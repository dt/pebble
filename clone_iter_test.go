// Copyright 2026 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package pebble_test

import (
	"bytes"
	"context"
	"testing"

	"github.com/cockroachdb/crlib/testutils/leaktest"
	"github.com/cockroachdb/pebble"
	"github.com/cockroachdb/pebble/cockroachkvs"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/rangekey"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
)

// TestVirtualClone_NewIterFirstSeesClonedData primary assertion is that
// NewIter().First() / Next() over the dst span returns the cloned data.
// This is the path CRDB's NewMVCCIterator and engine.NewIter use; Get-based
// assertions (which use the prefix-iter path) can succeed even when this
// path silently elides the cloned virtual SST.
func TestVirtualClone_NewIterFirstSeesClonedData(t *testing.T) {
	defer leaktest.AfterTest(t)()

	mem := vfs.NewMem()
	opts := &pebble.Options{
		Comparer:                    &cockroachkvs.Comparer,
		BlockPropertyCollectors:     cockroachkvs.BlockPropertyCollectors,
		FormatMajorVersion:          pebble.FormatPrefixSubstitution,
		FS:                          mem,
		KeySchema:                   cockroachkvs.KeySchema.Name,
		KeySchemas:                  sstable.MakeKeySchemas(&cockroachkvs.KeySchema),
		DisableAutomaticCompactions: true,
	}
	d, err := pebble.Open("", opts)
	require.NoError(t, err)
	defer func() { require.NoError(t, d.Close()) }()

	enc := func(roachKey []byte) []byte {
		return cockroachkvs.EncodeKey(nil, roachKey, nil)
	}
	encMVCC := func(roachKey []byte, walltime uint64) []byte {
		ver := []byte{
			byte(walltime >> 56), byte(walltime >> 48), byte(walltime >> 40), byte(walltime >> 32),
			byte(walltime >> 24), byte(walltime >> 16), byte(walltime >> 8), byte(walltime),
		}
		return cockroachkvs.EncodeKey(nil, roachKey, ver)
	}

	srcPrefix := []byte{0xfe, 0x8b}
	dstPrefix := []byte{0xfe, 0x8c}
	srcRoach := func(s string) []byte {
		return append(append([]byte{}, srcPrefix...), s...)
	}

	// 3 SETs + source rangedel + compact (this combo failed earlier).
	require.NoError(t, d.DeleteRange(
		enc(srcRoach("mmm")), enc([]byte{0xfe, 0x8c}), nil))
	for _, rk := range []string{"ppp", "qqq", "rrr"} {
		require.NoError(t, d.Set(encMVCC(srcRoach(rk), 100), []byte("v-"+rk), nil))
	}
	require.NoError(t, d.Flush())
	require.NoError(t, d.Compact(context.Background(),
		enc(srcRoach("aaa")), enc(srcRoach("zzz")), true))

	// Clone /Tenant/3/mmm..</Tenant/4) → /Tenant/4/mmm..</Tenant/5).
	srcSpan := pebble.KeyRange{
		Start: enc(srcRoach("mmm")),
		End:   enc([]byte{0xfe, 0x8c}),
	}
	dstSpan := pebble.KeyRange{
		Start: enc(append(append([]byte{}, dstPrefix...), "mmm"...)),
		End:   enc([]byte{0xfe, 0x8d}),
	}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstSpan, dstPrefix))

	// Dump SST metadata for visibility — what does HasPointKeys say?
	sstables, err := d.SSTables()
	require.NoError(t, err)
	for level, files := range sstables {
		for _, f := range files {
			t.Logf("L%d file=%d virtual=%v backing=%d size=%d "+
				"smallest=%x largest=%x seqnums=[%d,%d]",
				level, f.TableInfo.FileNum, f.Virtual,
				f.BackingSSTNum, f.Size,
				f.Smallest.UserKey, f.Largest.UserKey,
				f.SmallestSeqNum, f.LargestSeqNum)
		}
	}

	// PRIMARY assertion: NewIter().First() iteration must surface the cloned
	// SETs. This is the v2-iter path that CRDB uses.
	it, err := d.NewIter(&pebble.IterOptions{
		LowerBound: dstSpan.Start,
		UpperBound: dstSpan.End,
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, it.Close()) }()
	var keys [][]byte
	for valid := it.First(); valid; valid = it.Next() {
		keys = append(keys, append([]byte(nil), it.Key()...))
	}
	for _, k := range keys {
		t.Logf("iter key=%x", k)
	}
	require.Greater(t, len(keys), 0,
		"NewIter().First() returned 0 keys — the cloned virtual SST is not being included in the iter's level composition")
}

// TestVirtualClone_ScanInternal_RangeKeyAtPrefixEnd is a CRDB-shape
// integration test for the post-clone ScanInternal path. The source SST
// contains a range-key fragment whose End is exactly srcPrefix.PrefixEnd()
// (encoded as `\xfe\x8c\x00` under cockroachkvs); the dstPrefix is
// deliberately non-adjacent to srcPrefix (`\xfe\x90` vs `\xfe\x8b`), so a
// byte-arithmetic substitution that produced `dstPrefix + tail` instead of
// `dstPrefix.PrefixEnd() + tail` would surface as a visibly wrong End
// (`\xfe\x90\x00` instead of `\xfe\x91\x00`).
//
// The keyspanIter boundary-translation fix in colblk/keyspan.go is
// directly unit-tested in
// TestKeyspanIter_BlockPrefixSubstitution_EndAtPrefixEnd; this test guards
// the end-to-end pipeline (ScanInternal over a clone produced from a CRDB-
// realistic source SST) against future regressions.
func TestVirtualClone_ScanInternal_RangeKeyAtPrefixEnd(t *testing.T) {
	defer leaktest.AfterTest(t)()

	mem := vfs.NewMem()
	opts := &pebble.Options{
		Comparer:                    &cockroachkvs.Comparer,
		BlockPropertyCollectors:     cockroachkvs.BlockPropertyCollectors,
		FormatMajorVersion:          pebble.FormatPrefixSubstitution,
		FS:                          mem,
		KeySchema:                   cockroachkvs.KeySchema.Name,
		KeySchemas:                  sstable.MakeKeySchemas(&cockroachkvs.KeySchema),
		DisableAutomaticCompactions: true,
	}
	d, err := pebble.Open("", opts)
	require.NoError(t, err)
	defer func() { require.NoError(t, d.Close()) }()

	enc := func(roachKey []byte) []byte {
		return cockroachkvs.EncodeKey(nil, roachKey, nil)
	}
	encMVCC := func(roachKey []byte, walltime uint64) []byte {
		ver := []byte{
			byte(walltime >> 56), byte(walltime >> 48), byte(walltime >> 40), byte(walltime >> 32),
			byte(walltime >> 24), byte(walltime >> 16), byte(walltime >> 8), byte(walltime),
		}
		return cockroachkvs.EncodeKey(nil, roachKey, ver)
	}

	srcPrefix := []byte{0xfe, 0x8b}
	dstPrefix := []byte{0xfe, 0x90} // intentionally non-adjacent to srcPrefix
	srcRoach := func(s string) []byte {
		return append(append([]byte{}, srcPrefix...), s...)
	}

	// Many point keys inside srcPrefix so the SST has multiple data blocks
	// and the straddler path produces a substitution-backed virtual SST
	// (rather than degenerating to a full boundary-rewrite physical).
	for i := 0; i < 2000; i++ {
		k := []byte{byte('a' + (i / 26 / 26 % 26)), byte('a' + (i / 26 % 26)), byte('a' + (i % 26))}
		require.NoError(t, d.Set(encMVCC(srcRoach(string(k)), 100), []byte("v"), nil))
	}
	// Range key whose End is the exact prefix boundary: enc([]byte{0xfe,0x8c})
	// = `\xfe\x8c\x00`, which is srcPrefix.PrefixEnd() + the cockroachkvs
	// suffix-length sentinel byte. This is the End shape that triggers the
	// keyspanIter assertion.
	require.NoError(t, d.RangeKeySet(
		enc(srcRoach("ra")),
		enc([]byte{0xfe, 0x8c}),
		nil, []byte("rk-value"), nil))
	require.NoError(t, d.Flush())
	require.NoError(t, d.Compact(context.Background(),
		enc(srcRoach("aaa")), enc(srcRoach("zzz")), true))

	// Clone /Tenant/?(\xfe\x8b)/..</Tenant/?(\xfe\x8c)) → dstPrefix region.
	srcSpan := pebble.KeyRange{
		Start: enc(srcRoach("a")),
		End:   enc([]byte{0xfe, 0x8c}),
	}
	dstSpan := pebble.KeyRange{
		Start: enc(append(append([]byte{}, dstPrefix...), "a"...)),
		End:   enc([]byte{0xfe, 0x91}),
	}
	require.NoError(t, d.VirtualClone(context.Background(),
		srcSpan, srcPrefix, dstSpan, dstPrefix))

	var visited []struct {
		start, end []byte
	}
	visitRangeKey := func(start, end []byte, _ []rangekey.Key) error {
		visited = append(visited, struct {
			start, end []byte
		}{
			start: append([]byte(nil), start...),
			end:   append([]byte(nil), end...),
		})
		return nil
	}
	visitPointKey := func(*base.InternalKey, base.LazyValue, pebble.IteratorLevel) error {
		return nil
	}
	visitRangeDel := func(start, end []byte, seqNum base.SeqNum) error { return nil }

	err = d.ScanInternal(context.Background(), pebble.ScanInternalOptions{
		IterOptions: pebble.IterOptions{
			LowerBound: dstSpan.Start,
			UpperBound: dstSpan.End,
			KeyTypes:   pebble.IterKeyTypePointsAndRanges,
		},
		VisitPointKey: visitPointKey,
		VisitRangeDel: visitRangeDel,
		VisitRangeKey: visitRangeKey,
	})
	require.NoError(t, err)

	require.Len(t, visited, 1, "expected one range key fragment in dst")
	wantStart := enc(append(append([]byte{}, dstPrefix...), "ra"...))
	wantEnd := enc([]byte{0xfe, 0x91}) // dstPrefix.PrefixEnd() + sentinel
	require.Truef(t, bytes.Equal(visited[0].start, wantStart),
		"start: got %x, want %x", visited[0].start, wantStart)
	require.Truef(t, bytes.Equal(visited[0].end, wantEnd),
		"end: got %x, want %x (off-by-one would be %x)",
		visited[0].end, wantEnd, enc(dstPrefix))
}

// TestVirtualClone_ScanInternal_RangeDelAtPrefixEnd reproduces a second
// boundary panic exposed once the keyspanIter substitution boundary fix
// (commit 31a11bbc) corrected the dst-space End for fragments terminating
// at srcPrefix.PrefixEnd(). Before that fix, the substitution emitted a
// silently-wrong End that happened to fall short of the standIn's
// inclusive-largest point key, so `keyspan.Truncate` never tripped its
// "inclusive upper bound inside span" assertion. With the End now
// correctly translated to dstPrefix.PrefixEnd() + sentinel, the
// rangedel emitted through the substitution iter on the straddler
// virtual standIn covers the standIn's largest point key — and the
// straddler standIn deliberately excludes range-deletion contributions
// from its bounds (clone.go's "do NOT extend the virtual SST's bounds
// to encompass in-span range-deletion / range-key fragments" comment),
// so the inclusive-largest point key sits inside an emitted rangedel
// span. `truncatingIter.nextSpanWithinBounds` panics at truncate.go:137.
//
// The CRDB-reported stack:
//
//	pebble/internal/keyspan.(*truncatingIter).nextSpanWithinBounds at truncate.go:137
//	pebble/internal/keyspan.(*truncatingIter).First at truncate.go:83
//	... LevelIter / MergingIter / InterleavingIter ...
//	pebble.(*scanInternalIterator).seekGE at scan_internal.go:1268
//	pebble.scanInternalImpl at scan_internal.go:948
//	pebble.(*DB).ScanInternal at scan_internal.go:141
func TestVirtualClone_ScanInternal_RangeDelAtPrefixEnd(t *testing.T) {
	defer leaktest.AfterTest(t)()

	mem := vfs.NewMem()
	opts := &pebble.Options{
		Comparer:                    &cockroachkvs.Comparer,
		BlockPropertyCollectors:     cockroachkvs.BlockPropertyCollectors,
		FormatMajorVersion:          pebble.FormatPrefixSubstitution,
		FS:                          mem,
		KeySchema:                   cockroachkvs.KeySchema.Name,
		KeySchemas:                  sstable.MakeKeySchemas(&cockroachkvs.KeySchema),
		DisableAutomaticCompactions: true,
	}
	d, err := pebble.Open("", opts)
	require.NoError(t, err)
	defer func() { require.NoError(t, d.Close()) }()

	enc := func(roachKey []byte) []byte {
		return cockroachkvs.EncodeKey(nil, roachKey, nil)
	}
	encMVCC := func(roachKey []byte, walltime uint64) []byte {
		ver := []byte{
			byte(walltime >> 56), byte(walltime >> 48), byte(walltime >> 40), byte(walltime >> 32),
			byte(walltime >> 24), byte(walltime >> 16), byte(walltime >> 8), byte(walltime),
		}
		return cockroachkvs.EncodeKey(nil, roachKey, ver)
	}

	// CRDB-realistic prefixes (adjacent tenant boundaries).
	srcPrefix := []byte{0xfe, 0x8b}
	dstPrefix := []byte{0xfe, 0x8c}
	srcRoach := func(s string) []byte {
		return append(append([]byte{}, srcPrefix...), s...)
	}

	// Build a single source SST containing both point keys spanning the
	// full 'a'..'z' first-letter range AND a rangedel from "mmm" to
	// srcPrefix.PrefixEnd(). Doing this in a single committed batch and
	// flushing to a small-enough SST keeps both in the same physical
	// table — the on-disk shape that the CRDB scenario produces.
	val := bytes.Repeat([]byte("v"), 256)
	// First write the point keys WITHOUT the rangedel and flush — Pebble's
	// flush would otherwise drop point keys covered by a same-memtable
	// rangedel, hiding the on-disk shape we want.
	for first := byte('a'); first <= 'z'; first++ {
		for second := byte(0x00); ; second++ {
			require.NoError(t, d.Set(
				encMVCC(srcRoach(string([]byte{first, second, 0x80})), 100),
				val, nil))
			if second == 0xff {
				break
			}
		}
	}
	require.NoError(t, d.Flush())
	// Pin the older point keys with a snapshot before introducing the
	// rangedel; otherwise the upcoming compaction would elide point keys
	// past mmm that the rangedel covers, again hiding the shape under
	// test.
	snap := d.NewSnapshot()
	defer func() { _ = snap.Close() }()
	require.NoError(t, d.DeleteRange(
		enc(srcRoach("mmm")), enc([]byte{0xfe, 0x8c}), nil))
	require.NoError(t, d.Flush())
	// Compact to merge the two L0 SSTs into a single L6 SST that contains
	// BOTH the point keys past mmm AND the rangedel ending at
	// srcPrefix.PrefixEnd() — the on-disk shape that the CRDB scenario
	// produces.
	require.NoError(t, d.Compact(context.Background(),
		enc(srcRoach("a")), enc(srcRoach("zzz")), true))

	srcSpan := pebble.KeyRange{
		Start: enc(srcRoach("a")),
		End:   enc([]byte{0xfe, 0x8c}),
	}
	dstSpan := pebble.KeyRange{
		Start: enc(append(append([]byte{}, dstPrefix...), "a"...)),
		End:   enc([]byte{0xfe, 0x8d}),
	}
	preClone, err := d.SSTables()
	require.NoError(t, err)
	for level, files := range preClone {
		for _, f := range files {
			t.Logf("pre-clone L%d file=%d virtual=%v backing=%d "+
				"smallest=%x largest=%x size=%d",
				level, f.TableInfo.FileNum, f.Virtual, f.BackingSSTNum,
				f.Smallest.UserKey, f.Largest.UserKey, f.Size)
		}
	}

	require.NoError(t, d.VirtualClone(context.Background(),
		srcSpan, srcPrefix, dstSpan, dstPrefix))

	postClone, err := d.SSTables()
	require.NoError(t, err)
	for level, files := range postClone {
		for _, f := range files {
			t.Logf("post-clone L%d file=%d virtual=%v backing=%d "+
				"smallest=%x largest=%x size=%d",
				level, f.TableInfo.FileNum, f.Virtual, f.BackingSSTNum,
				f.Smallest.UserKey, f.Largest.UserKey, f.Size)
		}
	}

	visitPointKey := func(*base.InternalKey, base.LazyValue, pebble.IteratorLevel) error {
		return nil
	}
	visitRangeDel := func(start, end []byte, seqNum base.SeqNum) error { return nil }
	visitRangeKey := func(start, end []byte, _ []rangekey.Key) error { return nil }

	// ScanInternal over dst space. The bug fires inside the rangedel
	// truncating iterator before any visit callback runs.
	err = d.ScanInternal(context.Background(), pebble.ScanInternalOptions{
		IterOptions: pebble.IterOptions{
			LowerBound: dstSpan.Start,
			UpperBound: dstSpan.End,
			KeyTypes:   pebble.IterKeyTypePointsAndRanges,
		},
		VisitPointKey: visitPointKey,
		VisitRangeDel: visitRangeDel,
		VisitRangeKey: visitRangeKey,
	})
	require.NoError(t, err)
}
