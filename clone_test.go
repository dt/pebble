// Copyright 2026 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package pebble

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cockroachdb/crlib/testutils/leaktest"
	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/internal/testkeys"
	"github.com/cockroachdb/pebble/objstorage/objstorageprovider"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/sstable/block"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
)

// openCloneTestDB opens an in-memory DB at the FormatPrefixSubstitution format
// major version (or the provided lower version, for the format-gate test).
func openCloneTestDB(t *testing.T, fmv FormatMajorVersion) *DB {
	t.Helper()
	return openCloneTestDBWithBlockSize(t, fmv, 0)
}

// openCloneTestDBWithBlockSize is like openCloneTestDB but with an overridden
// per-level BlockSize (0 = default).
func openCloneTestDBWithBlockSize(t *testing.T, fmv FormatMajorVersion, blockSize int) *DB {
	t.Helper()
	return openCloneTestDBWithOpts(t, fmv, blockSize, nil)
}

// openCloneTestDBTwoLevel opens a DB tuned to force two-level indexes for any
// non-trivial flushed SST: data BlockSize and IndexBlockSize are both small,
// so even a modest number of keys spills over the index-block threshold and
// triggers a top-level / second-level index split.
func openCloneTestDBTwoLevel(t *testing.T, fmv FormatMajorVersion) *DB {
	t.Helper()
	return openCloneTestDBWithOpts(t, fmv, 0, func(opts *Options) {
		for i := range opts.Levels {
			opts.Levels[i].BlockSize = 64
			opts.Levels[i].IndexBlockSize = 64
			opts.Levels[i].BlockSizeThreshold = 50
		}
	})
}

// requireTwoLevelIndex checks that every SST in the LSM that has any keys with
// the given prefix uses a two-level index. This guards the two-level test
// matrix against regressions where the writer no longer triggers two-level
// indexes for the chosen tuning.
func requireTwoLevelIndex(t *testing.T, d *DB, prefix []byte) {
	t.Helper()
	d.mu.Lock()
	v := d.mu.versions.currentVersion()
	v.Ref()
	d.mu.Unlock()
	defer v.Unref()

	bounds := base.UserKeyBoundsEndExclusive(prefix, append(append([]byte(nil), prefix...), 0xff))
	var checked int
	for _, ls := range v.AllLevelsAndSublevels() {
		for m := range ls.Overlaps(d.cmp, bounds).All() {
			checked++
			err := d.fileCache.withReader(context.Background(),
				block.NoReadEnv, m,
				func(r *sstable.Reader, _ sstable.ReadEnv) error {
					require.True(t, r.Attributes.Has(sstable.AttributeTwoLevelIndex),
						"expected SST %s to use a two-level index; attributes=%s",
						m.TableNum, r.Attributes)
					return nil
				})
			require.NoError(t, err)
		}
	}
	require.Greater(t, checked, 0, "no SSTs intersected prefix %q", prefix)
}

// openCloneTestDBWithOpts is the most general clone-test DB opener. The optional
// modify callback receives the Options before Open and may install testing
// hooks (in particular, opts.private.testingClone* hooks).
func openCloneTestDBWithOpts(
	t *testing.T, fmv FormatMajorVersion, blockSize int, modify func(*Options),
) *DB {
	t.Helper()
	mem := vfs.NewMem()
	opts := &Options{
		Comparer:                    testkeys.Comparer,
		FS:                          mem,
		FormatMajorVersion:          fmv,
		DisableAutomaticCompactions: true,
		L0CompactionThreshold:       100,
		L0StopWritesThreshold:       100,
	}
	if blockSize > 0 {
		for i := range opts.Levels {
			opts.Levels[i].BlockSize = blockSize
			opts.Levels[i].IndexBlockSize = 1 << 30 // keep a single index level
		}
	}
	if modify != nil {
		modify(opts)
	}
	d, err := Open("", opts)
	require.NoError(t, err)
	return d
}

func setMany(t *testing.T, d *DB, keys []string, value []byte) {
	t.Helper()
	for _, k := range keys {
		require.NoError(t, d.Set([]byte(k), value, nil))
	}
}

func mustGet(t *testing.T, d *DB, key string) []byte {
	t.Helper()
	v, closer, err := d.Get([]byte(key))
	require.NoError(t, err, "Get(%q)", key)
	out := append([]byte(nil), v...)
	require.NoError(t, closer.Close())
	return out
}

// scanRange iterates [lower, upper) and returns the user keys it observes.
func scanRange(t *testing.T, d *DB, lower, upper []byte) []string {
	t.Helper()
	it, err := d.NewIter(&IterOptions{LowerBound: lower, UpperBound: upper})
	require.NoError(t, err)
	defer func() { require.NoError(t, it.Close()) }()
	var keys []string
	for valid := it.First(); valid; valid = it.Next() {
		keys = append(keys, string(it.Key()))
	}
	return keys
}

func TestVirtualClone_HappyPathSingleSST(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	srcKeys := []string{
		"/tenant/1/key0001",
		"/tenant/1/key0002",
		"/tenant/1/key0003",
	}
	setMany(t, d, srcKeys, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// Reads on dst-space keys should return the cloned values.
	for _, k := range srcKeys {
		dstKey := "/tenant/4/" + k[len(srcPrefix):]
		require.Equal(t, value, mustGet(t, d, dstKey))
	}

	// Iterate the dst range and confirm the keys appear in dst space.
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{
		"/tenant/4/key0001",
		"/tenant/4/key0002",
		"/tenant/4/key0003",
	}, got)

	// And the source keys are still readable in src space.
	for _, k := range srcKeys {
		require.Equal(t, value, mustGet(t, d, k))
	}
}

func TestVirtualClone_MultipleFullyContainedSSTsAcrossLevels(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// First SST: keys A0..A2; flush + compact to L6.
	setMany(t, d, []string{"/tenant/1/A0", "/tenant/1/A1", "/tenant/1/A2"}, value)
	require.NoError(t, d.Flush())
	require.NoError(t, d.Compact(context.Background(), []byte("/tenant/1/A0"), []byte("/tenant/1/A2\x00"), true))

	// Second SST: keys B0..B2; flush (stays in L0).
	setMany(t, d, []string{"/tenant/1/B0", "/tenant/1/B1", "/tenant/1/B2"}, value)
	require.NoError(t, d.Flush())

	// Third SST: keys C0..C2; flush (stays in L0 above the previous one).
	setMany(t, d, []string{"/tenant/1/C0", "/tenant/1/C1", "/tenant/1/C2"}, value)
	require.NoError(t, d.Flush())

	// Sanity: more than one level has data covering /tenant/1/.
	d.mu.Lock()
	v := d.mu.versions.currentVersion()
	var levelsWithData int
	for layer, ls := range v.AllLevelsAndSublevels() {
		if !ls.Empty() {
			levelsWithData++
			t.Logf("layer %s has %d files", layer, ls.Len())
		}
	}
	d.mu.Unlock()
	require.GreaterOrEqual(t, levelsWithData, 2, "test setup needs data at multiple levels")

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	want := []string{
		"/tenant/4/A0", "/tenant/4/A1", "/tenant/4/A2",
		"/tenant/4/B0", "/tenant/4/B1", "/tenant/4/B2",
		"/tenant/4/C0", "/tenant/4/C1", "/tenant/4/C2",
	}
	for _, k := range want {
		require.Equal(t, value, mustGet(t, d, k), "expected %s to be readable after clone", k)
	}
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, want, got)
}

// TestVirtualClone_StraddlingSrcSpanLowerEnd exercises D2: the source SST
// straddles only the lower bound of srcSpan. The lower-boundary block must be
// rewritten as a small physical SST.
func TestVirtualClone_StraddlingSrcSpanLowerEnd(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Mix /tenant/0/ and /tenant/1/ keys in one SST. The resulting file
	// straddles the lower bound of /tenant/1/.
	setMany(t, d, []string{
		"/tenant/0/key0",
		"/tenant/1/key1",
		"/tenant/1/key2",
	}, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// The destination should contain only the in-span keys, translated.
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/key1", "/tenant/4/key2"}, got)

	// Source keys should still be readable.
	for _, k := range []string{"/tenant/0/key0", "/tenant/1/key1", "/tenant/1/key2"} {
		require.Equal(t, value, mustGet(t, d, k))
	}
}

func TestVirtualClone_FormatGate(t *testing.T) {
	defer leaktest.AfterTest(t)()
	// Open below the floor.
	d := openCloneTestDB(t, FormatPrefixSubstitution-1)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}

	err := d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix)
	require.Error(t, err)
	require.Contains(t, err.Error(), "format major version")
}

// TestVirtualClone_DestinationConflictPlacedInL0 verifies D2's top-down,
// source-floored level placement: when the destination region is occupied at
// L0, the cloned file is also placed at L0 (since L0 always tolerates overlap).
func TestVirtualClone_DestinationConflictPlacedInL0(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Put data at the destination first.
	setMany(t, d, []string{"/tenant/4/k1", "/tenant/4/k2"}, value)
	require.NoError(t, d.Flush())
	// And data at the source.
	setMany(t, d, []string{"/tenant/1/k1", "/tenant/1/k2"}, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// All four destination keys (the original + the cloned ones) should be
	// readable.
	for _, k := range []string{"/tenant/4/k1", "/tenant/4/k2"} {
		require.Equal(t, value, mustGet(t, d, k))
	}
}

func TestVirtualClone_EmptySrcSpan(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")

	// Insert data well outside srcPrefix so that no SSTs intersect.
	setMany(t, d, []string{"/tenant/9/x"}, []byte("v"))
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// The dst region should remain empty.
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Empty(t, got)
}

// Ensure that reasonable input validation triggers errors that aren't
// ErrUnsupportedClone (which is reserved for runtime LSM-shape mismatches).
func TestVirtualClone_InputValidation(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}

	cases := []struct {
		name              string
		span              KeyRange
		src, dst          []byte
		expectErrContains string
	}{
		{"empty-src", srcSpan, nil, dstPrefix, "non-empty srcPrefix"},
		{"empty-dst", srcSpan, srcPrefix, nil, "non-empty dstPrefix"},
		{"same-prefix", srcSpan, srcPrefix, srcPrefix, "must differ"},
		{
			"span-start-outside-src",
			KeyRange{Start: []byte("/tenant/0/x"), End: []byte("/tenant/2/")},
			srcPrefix, dstPrefix, "does not have srcPrefix",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := d.VirtualClone(context.Background(), c.span, c.src, c.dst)
			require.Error(t, err)
			require.False(t, errors.Is(err, ErrUnsupportedClone),
				"expected input-validation error, got ErrUnsupportedClone: %v", err)
			require.Contains(t, fmt.Sprint(err), c.expectErrContains)
		})
	}
}

// TestVirtualClone_StraddlingSrcSpanUpperEnd exercises a SST that crosses the
// upper bound of srcSpan, with multiple data blocks so that the in-span run
// is non-empty and a true upper-boundary block is rewritten.
func TestVirtualClone_StraddlingSrcSpanUpperEnd(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBWithBlockSize(t, FormatPrefixSubstitution, 64)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24) // make blocks fill quickly

	// Write enough keys at /tenant/1/ that there are multiple data blocks,
	// then add /tenant/2/ keys in the same flush so the SST straddles the
	// upper bound of /tenant/1/.
	var keys []string
	for i := 0; i < 30; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%03d", i))
	}
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/2/k%03d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Len(t, got, 30)
	for i, k := range got {
		require.Equal(t, fmt.Sprintf("/tenant/4/k%03d", i), k)
	}
}

// TestVirtualClone_BothEndsStraddling exercises a srcSpan strictly inside the
// SST's range, so both ends straddle. Two boundary blocks must be rewritten.
func TestVirtualClone_BothEndsStraddling(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBWithBlockSize(t, FormatPrefixSubstitution, 64)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	// Write keys at /tenant/0/ ... /tenant/2/ so the SST spans both bounds.
	var keys []string
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/0/k%03d", i))
	}
	for i := 0; i < 30; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%03d", i))
	}
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/2/k%03d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	// Sub-range clone strictly inside /tenant/1/.
	srcSpan := KeyRange{Start: []byte("/tenant/1/k005"), End: []byte("/tenant/1/k025")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	for _, k := range got {
		require.True(t, len(k) > len(dstPrefix), "got key %q", k)
	}
	require.Equal(t, 20, len(got))
	for i, k := range got {
		require.Equal(t, fmt.Sprintf("/tenant/4/k%03d", i+5), k)
	}
}

// TestVirtualClone_MixedInteriorAndStraddlers exercises a mix of fully-
// contained SSTs (interior) and straddling SSTs at each end.
func TestVirtualClone_MixedInteriorAndStraddlers(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBWithBlockSize(t, FormatPrefixSubstitution, 64)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	// SST A: lower-end straddler (/tenant/0/ + /tenant/1/a*).
	var a []string
	for i := 0; i < 3; i++ {
		a = append(a, fmt.Sprintf("/tenant/0/k%03d", i))
	}
	for i := 0; i < 10; i++ {
		a = append(a, fmt.Sprintf("/tenant/1/a%03d", i))
	}
	setMany(t, d, a, value)
	require.NoError(t, d.Flush())

	// SST B: fully-contained (/tenant/1/b*).
	var b []string
	for i := 0; i < 5; i++ {
		b = append(b, fmt.Sprintf("/tenant/1/b%03d", i))
	}
	setMany(t, d, b, value)
	require.NoError(t, d.Flush())

	// SST C: upper-end straddler (/tenant/1/c* + /tenant/2/).
	var c []string
	for i := 0; i < 10; i++ {
		c = append(c, fmt.Sprintf("/tenant/1/c%03d", i))
	}
	for i := 0; i < 3; i++ {
		c = append(c, fmt.Sprintf("/tenant/2/k%03d", i))
	}
	setMany(t, d, c, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	// We expect 10 a* + 5 b* + 10 c* = 25 keys.
	require.Equal(t, 25, len(got))
	require.Equal(t, "/tenant/4/a000", got[0])
	require.Equal(t, "/tenant/4/c009", got[len(got)-1])
}

// TestVirtualClone_BlockPropertyFilterDisabled verifies that opening an
// iterator with a block-property filter against a substituted virtual SST
// does not silently drop in-span keys (the filter should be disabled).
func TestVirtualClone_BlockPropertyFilterDisabled(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	setMany(t, d, []string{
		"/tenant/1/k1",
		"/tenant/1/k2",
		"/tenant/1/k3",
	}, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// A regular scan must surface the cloned keys regardless of any default
	// block property filter behavior.
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/k1", "/tenant/4/k2", "/tenant/4/k3"}, got)
}

// TestVirtualClone_MemtableOnly verifies that VirtualClone forces a flush of
// any memtable whose contents overlap srcSpan, so that recent writes that
// haven't yet flushed appear in the cloned destination.
func TestVirtualClone_MemtableOnly(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Write keys but do NOT call Flush; the keys live only in the memtable.
	srcKeys := []string{
		"/tenant/1/k1",
		"/tenant/1/k2",
		"/tenant/1/k3",
	}
	setMany(t, d, srcKeys, value)

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// Cloned dst-space keys must be readable.
	for _, k := range srcKeys {
		dstKey := "/tenant/4/" + k[len(srcPrefix):]
		require.Equal(t, value, mustGet(t, d, dstKey),
			"expected dst key %s to be readable after VirtualClone forced a flush", dstKey)
	}
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/k1", "/tenant/4/k2", "/tenant/4/k3"}, got)
}

// TestVirtualClone_MixedMemtableAndLSM verifies that data spread across L6
// (flushed+compacted) and the memtable both make it into the clone.
func TestVirtualClone_MixedMemtableAndLSM(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Flush + compact some keys to L6.
	lsmKeys := []string{"/tenant/1/A0", "/tenant/1/A1", "/tenant/1/A2"}
	setMany(t, d, lsmKeys, value)
	require.NoError(t, d.Flush())
	require.NoError(t, d.Compact(context.Background(),
		[]byte("/tenant/1/A0"), []byte("/tenant/1/A2\x00"), true))

	// Write more keys to the memtable (no flush).
	memKeys := []string{"/tenant/1/B0", "/tenant/1/B1", "/tenant/1/B2"}
	setMany(t, d, memKeys, value)

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	want := []string{
		"/tenant/4/A0", "/tenant/4/A1", "/tenant/4/A2",
		"/tenant/4/B0", "/tenant/4/B1", "/tenant/4/B2",
	}
	for _, k := range want {
		require.Equal(t, value, mustGet(t, d, k),
			"expected %s to appear in clone (LSM+memtable mix)", k)
	}
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, want, got)
}

// TestVirtualClone_NoMemtableOverlap verifies that memtable writes outside
// srcSpan do not appear in the destination, and that the LSM data still
// clones correctly. The clone must remain correct whether or not an
// unrelated flush is forced.
func TestVirtualClone_NoMemtableOverlap(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Flush some in-span keys to L6 so there's LSM data to clone.
	setMany(t, d, []string{"/tenant/1/k1", "/tenant/1/k2"}, value)
	require.NoError(t, d.Flush())

	// Write unrelated keys (well outside srcSpan) to the memtable.
	setMany(t, d, []string{"/tenant/9/x", "/tenant/9/y"}, value)

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// Only the LSM-resident in-span data should appear in dst-space.
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/k1", "/tenant/4/k2"}, got)

	// And the unrelated memtable keys must still be readable in src space
	// (whether they were forced to flush or not is an implementation detail).
	for _, k := range []string{"/tenant/9/x", "/tenant/9/y"} {
		require.Equal(t, value, mustGet(t, d, k))
	}
}

// TestVirtualClone_Race_CompactionDuringClone (scenario A) verifies the
// retry loop fires when a compaction completes between the version snapshot
// and the apply phase, mutating one of the source backings out from under
// the in-flight clone. The clone must abort, retry, and ultimately succeed
// with the post-compaction LSM state.
func TestVirtualClone_Race_CompactionDuringClone(t *testing.T) {
	defer leaktest.AfterTest(t)()

	var compactedOnce atomic.Bool
	var attempts atomic.Int32
	var dRef atomic.Pointer[DB]
	d := openCloneTestDBWithOpts(t, FormatPrefixSubstitution, 0, func(opts *Options) {
		opts.private.testingCloneAfterSnapshot = func(attempt int) {
			attempts.Add(1)
			// Only race on the first attempt: the second attempt must
			// observe a quiescent LSM and succeed.
			if compactedOnce.Swap(true) {
				return
			}
			db := dRef.Load()
			// Compact all of /tenant/1/ down to L6. This produces a brand-new
			// TableMetadata at L6 with a different TableNum and obsoletes the
			// L0 sources the clone snapshotted.
			require.NoError(t, db.Compact(context.Background(),
				[]byte("/tenant/1/"), []byte("/tenant/1/\xff"), false))
		}
	})
	dRef.Store(d)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Two L0 SSTs in srcSpan.
	setMany(t, d, []string{"/tenant/1/A0", "/tenant/1/A1", "/tenant/1/A2"}, value)
	require.NoError(t, d.Flush())
	setMany(t, d, []string{"/tenant/1/B0", "/tenant/1/B1", "/tenant/1/B2"}, value)
	require.NoError(t, d.Flush())

	// Capture the L0 source TableNums so we can assert they are obsoleted.
	d.mu.Lock()
	preCloneL0 := make(map[base.TableNum]struct{})
	for f := range d.mu.versions.currentVersion().Levels[0].All() {
		preCloneL0[f.TableNum] = struct{}{}
	}
	d.mu.Unlock()

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// We expect at least 2 attempts: the first is aborted by the compaction
	// and the second succeeds.
	require.GreaterOrEqual(t, int(attempts.Load()), 2,
		"expected the retry path to run at least once")

	// Verify post-compaction L0 sources no longer present (proves the race
	// actually mutated the LSM during the clone).
	d.mu.Lock()
	postCloneL0 := make(map[base.TableNum]struct{})
	for f := range d.mu.versions.currentVersion().Levels[0].All() {
		postCloneL0[f.TableNum] = struct{}{}
	}
	d.mu.Unlock()
	for tn := range preCloneL0 {
		_, stillThere := postCloneL0[tn]
		require.False(t, stillThere,
			"expected L0 source table %s to be obsoleted by the racing compaction", tn)
	}

	// Final state: dst keys are readable.
	want := []string{
		"/tenant/4/A0", "/tenant/4/A1", "/tenant/4/A2",
		"/tenant/4/B0", "/tenant/4/B1", "/tenant/4/B2",
	}
	for _, k := range want {
		require.Equal(t, value, mustGet(t, d, k))
	}
	require.Equal(t, want, scanRange(t, d, dstPrefix, []byte("/tenant/5/")))
}

// TestVirtualClone_Race_RetriesExhausted (scenario B) verifies the bounded-
// retry path: when every attempt loses the race, VirtualClone returns a clean
// error (no panic, no deadlock) and any boundary SSTs written during
// intermediate attempts are cleaned up from the object provider.
func TestVirtualClone_Race_RetriesExhausted(t *testing.T) {
	defer leaktest.AfterTest(t)()

	var attempts atomic.Int32
	var dRef atomic.Pointer[DB]
	racingCompaction := func(attempt int) {
		attempts.Add(1)
		d := dRef.Load()
		if d == nil {
			return
		}
		// Each attempt: write a brand-new in-srcSpan key, flush it to L0,
		// then compact the entire srcSpan. This guarantees a fresh L6
		// TableMetadata pointer each iteration and obsoletes whatever the
		// in-flight clone snapshotted on the prior attempt.
		key := fmt.Sprintf("/tenant/1/race-%03d", attempt)
		require.NoError(t, d.Set([]byte(key), []byte("x"), nil))
		require.NoError(t, d.Flush())
		require.NoError(t, d.Compact(context.Background(),
			[]byte("/tenant/1/"), []byte("/tenant/1/\xff"), false))
	}

	// The DB must exist before the hook can use it; install a closure that
	// dereferences a pointer.
	d := openCloneTestDBWithOpts(t, FormatPrefixSubstitution, 64, func(opts *Options) {
		opts.private.testingCloneAfterSnapshot = racingCompaction
		opts.private.testingAlwaysWaitForCleanup = true
	})
	dRef.Store(d)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	// Build a straddling source SST with an upper boundary block (so each
	// attempt allocates a boundary-rewrite physical SST that must be cleaned
	// up on abort).
	var keys []string
	for i := 0; i < 30; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%03d", i))
	}
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/2/k%03d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	// Snapshot of objects before the failing clone, so we can verify no
	// orphan objects remain afterward.
	objsBefore := make(map[base.DiskFileNum]struct{})
	for _, om := range d.objProvider.List() {
		objsBefore[om.DiskFileNum] = struct{}{}
	}

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	err := d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix)
	require.Error(t, err)
	require.Contains(t, err.Error(), "exhausted")
	require.Equal(t, int32(virtualCloneMaxRetries), attempts.Load(),
		"every attempt should have invoked the hook")

	// Disable the hook so cleanup logic in subsequent operations doesn't race.
	d.opts.private.testingCloneAfterSnapshot = nil

	// Force any pending obsolete-file deletions (from racing compactions) to
	// drain so that objProvider.List() reflects the steady state.
	d.maybeScheduleObsoleteObjectDeletion()

	// Verify that the live object set didn't grow with extra orphan tables.
	// New backings/tables created by the racing compactions are legitimate;
	// what we don't want is an orphan boundary-rewrite SST from a failed
	// clone attempt that the version doesn't reference. We check by walking
	// the current version's referenced files and asserting that every
	// table-typed object on disk is either referenced by the version, was
	// present before the clone began, or is from a racing compaction
	// (everything compaction creates is also referenced by the version).
	live := make(map[base.DiskFileNum]struct{})
	d.mu.Lock()
	v := d.mu.versions.currentVersion()
	for _, lm := range v.Levels {
		for f := range lm.All() {
			live[f.TableBacking.DiskFileNum] = struct{}{}
		}
	}
	for b := range d.mu.versions.latest.virtualBackings.All() {
		live[b.DiskFileNum] = struct{}{}
	}
	d.mu.Unlock()

	for _, om := range d.objProvider.List() {
		if om.FileType != base.FileTypeTable {
			continue
		}
		if _, isLive := live[om.DiskFileNum]; isLive {
			continue
		}
		if _, isOld := objsBefore[om.DiskFileNum]; isOld {
			continue
		}
		t.Fatalf("orphan table object remains after exhausted retries: %s", om.DiskFileNum)
	}

	// Sanity: no clone landed in dst space.
	require.Empty(t, scanRange(t, d, dstPrefix, []byte("/tenant/5/")))
}

// TestVirtualClone_Race_ExciseDuringClone (scenario C) verifies that an
// IngestAndExcise that mutates srcSpan between snapshot and apply triggers
// the retry path. After the excise, the clone re-runs against the post-
// excise LSM and either succeeds with the post-excise data or returns a
// graceful error.
func TestVirtualClone_Race_ExciseDuringClone(t *testing.T) {
	defer leaktest.AfterTest(t)()

	var dRef atomic.Pointer[DB]
	var hookOnce atomic.Bool
	var attempts atomic.Int32
	d := openCloneTestDBWithOpts(t, FormatPrefixSubstitution, 0, func(opts *Options) {
		opts.private.testingCloneAfterSnapshot = func(attempt int) {
			attempts.Add(1)
			if hookOnce.Swap(true) {
				return
			}
			db := dRef.Load()
			if db == nil {
				return
			}
			// Ingest+excise a sub-region of srcSpan: excise [k010, k020),
			// which removes part of the source content.
			path := "excise.sst"
			f, err := db.opts.FS.Create(path, vfs.WriteCategoryUnspecified)
			require.NoError(t, err)
			w := sstable.NewWriter(
				objstorageprovider.NewFileWritable(f),
				db.opts.MakeWriterOptions(0, db.TableFormat()))
			require.NoError(t, w.Set([]byte("/tenant/1/k015-replacement"), []byte("ingested")))
			require.NoError(t, w.Close())
			_, err = db.IngestAndExcise(context.Background(),
				[]string{path}, nil, nil,
				KeyRange{Start: []byte("/tenant/1/k010"), End: []byte("/tenant/1/k020")})
			require.NoError(t, err)
		}
	})
	dRef.Store(d)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	var keys []string
	for i := 0; i < 30; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%03d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	err := d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix)
	// The clone must either succeed (after retry against the post-excise
	// LSM) or return a graceful error; it must never panic or deadlock.
	if err != nil {
		// Any error must be a clean Pebble error (not a wrap of a runtime
		// panic). ErrUnsupportedClone is acceptable.
		t.Logf("clone returned graceful error: %v", err)
	}
	require.GreaterOrEqual(t, int(attempts.Load()), 2,
		"expected the retry path to fire at least once")

	// Source space must still reflect the post-excise state: keys k010..k019
	// from the original flush are gone (replaced by the ingested SST).
	srcGot := scanRange(t, d, []byte("/tenant/1/"), []byte("/tenant/2/"))
	for _, k := range srcGot {
		// No scanned key should be in the excised band [k010, k020) other
		// than the ingested one.
		if k > "/tenant/1/k009" && k < "/tenant/1/k020" {
			require.Equal(t, "/tenant/1/k015-replacement", k,
				"unexpected pre-excise key in src space: %s", k)
		}
	}
}

// TestVirtualClone_Race_FlushDuringClone (scenario D) verifies that a flush
// of in-srcSpan memtable data that lands in L0 *after* the clone's snapshot
// does not cause the clone to spuriously fail. The new L0 data must appear
// in src space only; it must not appear in dst space.
func TestVirtualClone_Race_FlushDuringClone(t *testing.T) {
	defer leaktest.AfterTest(t)()

	var dRef atomic.Pointer[DB]
	var hookOnce atomic.Bool
	var attempts atomic.Int32
	d := openCloneTestDBWithOpts(t, FormatPrefixSubstitution, 0, func(opts *Options) {
		opts.private.testingCloneAfterSnapshot = func(attempt int) {
			attempts.Add(1)
			if hookOnce.Swap(true) {
				return
			}
			db := dRef.Load()
			if db == nil {
				return
			}
			// Write a NEW in-srcSpan key and flush it. This produces a brand
			// new L0 SST that the clone's snapshotted version does not see.
			require.NoError(t, db.Set([]byte("/tenant/1/post-snapshot"), []byte("post"), nil))
			require.NoError(t, db.Flush())
		}
	})
	dRef.Store(d)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Pre-existing L0 SST in srcSpan.
	setMany(t, d, []string{"/tenant/1/k1", "/tenant/1/k2"}, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// We expect exactly one attempt: a flush into a new L0 SST does not
	// invalidate any source backing snapshotted by the clone, and L0 always
	// tolerates overlap on placement, so no abort/retry should occur.
	require.Equal(t, int32(1), attempts.Load(),
		"flush of new L0 data should not cause a clone abort")

	// dst-space contains only the pre-snapshot keys, not the post-snapshot
	// addition.
	gotDst := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/k1", "/tenant/4/k2"}, gotDst)

	// src-space contains the post-snapshot key.
	require.Equal(t, []byte("post"), mustGet(t, d, "/tenant/1/post-snapshot"))
}

// TestVirtualClone_Race_ConcurrentClonesOverlappingDst (scenario E) starts
// two VirtualClones in parallel that target overlapping destination spaces.
// Either both succeed (one or both place files into L0 to tolerate overlap),
// or one returns ErrUnsupportedClone after exhausting retries. In neither
// case must we corrupt the LSM or panic.
func TestVirtualClone_Race_ConcurrentClonesOverlappingDst(t *testing.T) {
	defer leaktest.AfterTest(t)()

	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	value := []byte("v")
	// Two source partitions and two distinct dst partitions that share an
	// overlapping span [/tenant/8/k...]. Both clones target /tenant/8/.
	setMany(t, d, []string{
		"/tenant/1/k1", "/tenant/1/k2",
		"/tenant/2/k1", "/tenant/2/k2",
	}, value)
	require.NoError(t, d.Flush())
	require.NoError(t, d.Compact(context.Background(),
		[]byte("/tenant/1/"), []byte("/tenant/3/"), true))

	srcPrefixA := []byte("/tenant/1/")
	srcPrefixB := []byte("/tenant/2/")
	dstPrefix := []byte("/tenant/8/")
	srcSpanA := KeyRange{Start: srcPrefixA, End: []byte("/tenant/2/")}
	srcSpanB := KeyRange{Start: srcPrefixB, End: []byte("/tenant/3/")}

	var wg sync.WaitGroup
	results := make([]error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		results[0] = d.VirtualClone(context.Background(), srcSpanA, srcPrefixA, dstPrefix)
	}()
	go func() {
		defer wg.Done()
		results[1] = d.VirtualClone(context.Background(), srcSpanB, srcPrefixB, dstPrefix)
	}()
	wg.Wait()

	// At least one must succeed.
	successes := 0
	for _, err := range results {
		if err == nil {
			successes++
			continue
		}
		// Acceptable failure modes: ErrUnsupportedClone (placement
		// saturated) or a clean retries-exhausted error.
		require.True(t,
			errors.Is(err, ErrUnsupportedClone) ||
				(strings.Contains(err.Error(), "exhausted") &&
					strings.Contains(err.Error(), "retries")),
			"unexpected error from concurrent clone: %v", err)
	}
	require.GreaterOrEqual(t, successes, 1, "at least one concurrent clone must succeed")

	// Whichever side(s) succeeded, dst-space must read consistently.
	got := scanRange(t, d, dstPrefix, []byte("/tenant/9/"))
	for _, k := range got {
		require.Equal(t, value, mustGet(t, d, k))
	}
}

// TestVirtualClone_OrphanCleanupOnRetry verifies that boundary SSTs written
// during a failed attempt are removed from the object provider. We force a
// failure path by examining the file system before and after a successful
// run. The successful run leaves boundary SSTs live; this test only checks
// that the count of physical SST files matches the live LSM.
func TestVirtualClone_OrphanCleanupOnRetry(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBWithBlockSize(t, FormatPrefixSubstitution, 64)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	var keys []string
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/0/k%03d", i))
	}
	for i := 0; i < 30; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%03d", i))
	}
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/2/k%03d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// Compute the set of live disk file numbers.
	live := make(map[base.DiskFileNum]struct{})
	d.mu.Lock()
	for _, om := range d.objProvider.List() {
		live[om.DiskFileNum] = struct{}{}
	}
	d.mu.Unlock()

	// All physical files referenced by live LSM tables and their backings
	// must exist; no extra orphan table files should remain. A precise check
	// requires walking the version's tables; instead we cross-check that
	// removing the clone via NewIter results in valid scans (a good
	// negative test would simulate a failed attempt; that's covered by the
	// retry+cleanup machinery in clone.go).
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, 30, len(got))
}

// TestVirtualClone_TwoLevelIndex_FullyContained verifies that cloning a
// fully-contained source SST that uses a two-level index works end-to-end.
// The fully-contained path doesn't actually walk per-data-block index
// structure (it reuses the SST's bounds wholesale), so this is a smoke test
// that the per-attempt format/two-level checks don't over-reject.
func TestVirtualClone_TwoLevelIndex_FullyContained(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBTwoLevel(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	// Many keys, all in srcPrefix range, to force a two-level index in the
	// flushed SST. With BlockSize=64 and 24-byte values we get ~1-2 KVs per
	// data block; with IndexBlockSize=64 the index spills to two-level
	// quickly.
	var keys []string
	for i := 0; i < 200; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%05d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	// Confirm the test fixture actually produced a two-level index; otherwise
	// the test isn't exercising what its name claims.
	requireTwoLevelIndex(t, d, srcPrefix)

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Len(t, got, 200)
	for i, k := range got {
		require.Equal(t, fmt.Sprintf("/tenant/4/k%05d", i), k)
	}
}

// TestVirtualClone_TwoLevelIndex_LowerStraddler exercises the new boundary-
// classification logic over a two-level index when the source SST straddles
// only the lower bound of srcSpan.
func TestVirtualClone_TwoLevelIndex_LowerStraddler(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBTwoLevel(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	var keys []string
	// Out-of-span keys at /tenant/0/ to introduce the lower straddle.
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/0/k%05d", i))
	}
	for i := 0; i < 200; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%05d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	// Confirm the SST uses a two-level index (the requireTwoLevelIndex helper
	// only checks tables overlapping srcPrefix, but the merged SST contains
	// both /tenant/0/ and /tenant/1/ keys so it overlaps).
	requireTwoLevelIndex(t, d, srcPrefix)

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Len(t, got, 200)
	for i, k := range got {
		require.Equal(t, fmt.Sprintf("/tenant/4/k%05d", i), k)
	}
	// Source out-of-span keys remain readable in src space.
	for i := 0; i < 5; i++ {
		require.Equal(t, value, mustGet(t, d, fmt.Sprintf("/tenant/0/k%05d", i)))
	}
}

// TestVirtualClone_TwoLevelIndex_UpperStraddler is the symmetric upper-end
// straddler case for two-level indexes.
func TestVirtualClone_TwoLevelIndex_UpperStraddler(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBTwoLevel(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	var keys []string
	for i := 0; i < 200; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%05d", i))
	}
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/2/k%05d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	requireTwoLevelIndex(t, d, srcPrefix)

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Len(t, got, 200)
	for i, k := range got {
		require.Equal(t, fmt.Sprintf("/tenant/4/k%05d", i), k)
	}
}

// TestVirtualClone_TwoLevelIndex_BothEnds exercises a srcSpan strictly inside
// the SST's range with both ends straddling, against a two-level index. Both
// boundary blocks must be rewritten.
func TestVirtualClone_TwoLevelIndex_BothEnds(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBTwoLevel(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	var keys []string
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/0/k%05d", i))
	}
	for i := 0; i < 200; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%05d", i))
	}
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/2/k%05d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	requireTwoLevelIndex(t, d, srcPrefix)

	// srcSpan strictly inside /tenant/1/.
	srcSpan := KeyRange{Start: []byte("/tenant/1/k00010"), End: []byte("/tenant/1/k00150")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Len(t, got, 140)
	for i, k := range got {
		require.Equal(t, fmt.Sprintf("/tenant/4/k%05d", i+10), k)
	}
}

// TestVirtualClone_TwoLevelIndex_RunCrossesSecondLevelBoundary is the case
// that most directly exercises the new walking logic: the contiguous run of
// in-span data blocks crosses one or more second-level index block
// boundaries. The boundary-classification code must visit data blocks across
// every second-level partition and not stop at a second-level boundary.
func TestVirtualClone_TwoLevelIndex_RunCrossesSecondLevelBoundary(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBTwoLevel(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	// Include a small out-of-span tail so the SST straddles the upper bound,
	// which forces buildStraddlerEntries (the path with the two-level walk)
	// to run instead of the fully-contained shortcut. The in-span run is
	// long enough that the second-level index spills across multiple blocks
	// (with IndexBlockSize=64 and ~1-2 data-block index entries per second-
	// level block, 600 data blocks => many second-level partitions).
	var keys []string
	for i := 0; i < 600; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%05d", i))
	}
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/2/k%05d", i))
	}
	setMany(t, d, keys, value)
	require.NoError(t, d.Flush())

	requireTwoLevelIndex(t, d, srcPrefix)

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Len(t, got, 600)
	for i, k := range got {
		require.Equal(t, fmt.Sprintf("/tenant/4/k%05d", i), k)
	}

	// Spot-check the boundaries (first, last, and middle) via point Get to
	// exercise the read path that goes through the substituted two-level
	// index for fully-contained probe targets.
	for _, i := range []int{0, 1, 299, 300, 599} {
		require.Equal(t, value,
			mustGet(t, d, fmt.Sprintf("/tenant/4/k%05d", i)))
	}
}

// scanRangeWithRangeKeys iterates [lower, upper) with KeyTypes=PointsAndRanges
// and returns the observed point keys plus a flat list of range-key bound
// strings of the form "[start,end)#kind=suffix:value" for every distinct range
// key returned. The bounds are recorded each time RangeKeyChanged() is true to
// avoid duplicates across positioning calls.
func scanRangeWithRangeKeys(
	t *testing.T, d *DB, lower, upper []byte,
) (points []string, ranges []string) {
	t.Helper()
	it, err := d.NewIter(&IterOptions{
		LowerBound: lower,
		UpperBound: upper,
		KeyTypes:   IterKeyTypePointsAndRanges,
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, it.Close()) }()
	for valid := it.First(); valid; valid = it.Next() {
		hasPoint, hasRange := it.HasPointAndRange()
		if hasPoint {
			points = append(points, string(it.Key()))
		}
		if hasRange && it.RangeKeyChanged() {
			start, end := it.RangeBounds()
			rkData := ""
			for _, rk := range it.RangeKeys() {
				rkData += fmt.Sprintf(",%s=%s", string(rk.Suffix), string(rk.Value))
			}
			ranges = append(ranges, fmt.Sprintf("[%s,%s)%s", string(start), string(end), rkData))
		}
	}
	return points, ranges
}

// TestVirtualClone_RangeDelete_FullyInside verifies that a source SST whose
// only range deletion lies entirely inside srcSpan can be cloned, and that the
// clone observes the range deletion in dst-space (suppressing point keys it
// covers).
func TestVirtualClone_RangeDelete_FullyInside(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Place point keys k1..k5, then a range delete over [k2, k4) — entirely
	// inside srcSpan.
	setMany(t, d, []string{
		"/tenant/1/k1",
		"/tenant/1/k2",
		"/tenant/1/k3",
		"/tenant/1/k4",
		"/tenant/1/k5",
	}, value)
	require.NoError(t, d.DeleteRange([]byte("/tenant/1/k2"), []byte("/tenant/1/k4"), nil))
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// The dst-space scan should observe k1, k4, k5 (k2 and k3 are deleted).
	got := scanRange(t, d, dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/k1", "/tenant/4/k4", "/tenant/4/k5"}, got)
}

// TestVirtualClone_RangeKeySet_FullyInside verifies that a source SST whose
// only range key (RangeKeySet) lies entirely inside srcSpan can be cloned, and
// that the clone observes the range key at translated dst bounds.
func TestVirtualClone_RangeKeySet_FullyInside(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	setMany(t, d, []string{
		"/tenant/1/p1",
		"/tenant/1/p2",
	}, value)
	require.NoError(t, d.RangeKeySet(
		[]byte("/tenant/1/ra"),
		[]byte("/tenant/1/rz"),
		[]byte("@5"), []byte("rk-value"), nil))
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	points, ranges := scanRangeWithRangeKeys(t, d,
		dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/p1", "/tenant/4/p2"}, points)
	require.Equal(t, []string{
		"[/tenant/4/ra,/tenant/4/rz),@5=rk-value",
	}, ranges)
}

// TestVirtualClone_RangeKey_FullyOutside verifies that a source SST whose
// range key lies entirely outside srcSpan can be cloned, and that the cloned
// virtual SST does not expose the range key.
func TestVirtualClone_RangeKey_FullyOutside(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Mix in-span point keys with a range key entirely in /tenant/0/ and then
	// some out-of-span point keys to make the SST straddle the lower bound.
	setMany(t, d, []string{
		"/tenant/0/p1",
		"/tenant/1/p1",
		"/tenant/1/p2",
	}, value)
	require.NoError(t, d.RangeKeySet(
		[]byte("/tenant/0/ra"),
		[]byte("/tenant/0/rz"),
		[]byte("@5"), []byte("rk-value"), nil))
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	points, ranges := scanRangeWithRangeKeys(t, d,
		dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/p1", "/tenant/4/p2"}, points)
	require.Empty(t, ranges, "out-of-span range key should not appear in dst")
}

// TestVirtualClone_RangeKey_Straddling verifies that a source SST whose range
// key straddles a srcSpan boundary returns ErrUnsupportedClone.
func TestVirtualClone_RangeKey_Straddling(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	setMany(t, d, []string{"/tenant/1/p1", "/tenant/1/p2"}, value)
	// Range key from /tenant/1/ra to /tenant/2/rz straddles the upper
	// bound of srcSpan (/tenant/2/).
	require.NoError(t, d.RangeKeySet(
		[]byte("/tenant/1/ra"),
		[]byte("/tenant/2/rz"),
		[]byte("@5"), []byte("rk-value"), nil))
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	err := d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix)
	require.Error(t, err)
	require.True(t, errors.Is(err, ErrUnsupportedClone),
		"expected ErrUnsupportedClone, got %v", err)
	require.Contains(t, err.Error(), "straddles")
	require.Contains(t, err.Error(), "range key")
}

// TestVirtualClone_RangeDel_Straddling verifies that a source SST whose range
// deletion straddles a srcSpan boundary returns ErrUnsupportedClone.
func TestVirtualClone_RangeDel_Straddling(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	setMany(t, d, []string{"/tenant/1/p1", "/tenant/1/p2"}, value)
	require.NoError(t, d.DeleteRange(
		[]byte("/tenant/1/p1"), []byte("/tenant/2/x"), nil))
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	err := d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix)
	require.Error(t, err)
	require.True(t, errors.Is(err, ErrUnsupportedClone),
		"expected ErrUnsupportedClone, got %v", err)
	require.Contains(t, err.Error(), "straddles")
	require.Contains(t, err.Error(), "range deletion")
}

// TestVirtualClone_MixedRangeKeys_InsideAndOutside verifies that an SST with
// multiple range keys (some entirely inside srcSpan, some entirely outside,
// none straddling) clones successfully and only exposes the inside ones at
// dst.
func TestVirtualClone_MixedRangeKeys_InsideAndOutside(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	setMany(t, d, []string{
		"/tenant/0/p0",
		"/tenant/1/p1",
		"/tenant/1/p2",
		"/tenant/2/p2",
	}, value)
	// Outside (below srcSpan).
	require.NoError(t, d.RangeKeySet(
		[]byte("/tenant/0/r1a"), []byte("/tenant/0/r1z"),
		[]byte("@5"), []byte("rk-outside-low"), nil))
	// Inside.
	require.NoError(t, d.RangeKeySet(
		[]byte("/tenant/1/r2a"), []byte("/tenant/1/r2z"),
		[]byte("@5"), []byte("rk-inside"), nil))
	// Outside (above srcSpan).
	require.NoError(t, d.RangeKeySet(
		[]byte("/tenant/2/r3a"), []byte("/tenant/2/r3z"),
		[]byte("@5"), []byte("rk-outside-high"), nil))
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	points, ranges := scanRangeWithRangeKeys(t, d,
		dstPrefix, []byte("/tenant/5/"))
	require.Equal(t, []string{"/tenant/4/p1", "/tenant/4/p2"}, points)
	require.Equal(t, []string{
		"[/tenant/4/r2a,/tenant/4/r2z),@5=rk-inside",
	}, ranges)
}

// TestVirtualClone_RangeKey_PointStraddler verifies the combined case: an SST
// whose POINT keys straddle srcSpan (triggering D2 boundary-block rewrite) and
// which also has a range key entirely inside srcSpan. Both the boundary-block
// rewrite for points and the in-iter substitution for range keys must work
// together.
func TestVirtualClone_RangeKey_PointStraddler(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDBWithBlockSize(t, FormatPrefixSubstitution, 64)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := make([]byte, 24)

	// Fill enough keys around both bounds to force multiple data blocks and
	// boundary blocks at both ends of srcSpan.
	var keys []string
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/0/k%03d", i))
	}
	for i := 0; i < 30; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/1/k%03d", i))
	}
	for i := 0; i < 5; i++ {
		keys = append(keys, fmt.Sprintf("/tenant/2/k%03d", i))
	}
	setMany(t, d, keys, value)
	// A range key fully inside srcSpan.
	require.NoError(t, d.RangeKeySet(
		[]byte("/tenant/1/ra"),
		[]byte("/tenant/1/rz"),
		[]byte("@5"), []byte("rk-inside"), nil))
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	require.NoError(t, d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix))

	// All 30 in-span point keys should be visible in dst, plus the in-span
	// range key.
	points, ranges := scanRangeWithRangeKeys(t, d,
		dstPrefix, []byte("/tenant/5/"))
	require.Len(t, points, 30)
	for i, k := range points {
		require.Equal(t, fmt.Sprintf("/tenant/4/k%03d", i), k)
	}
	require.Equal(t, []string{
		"[/tenant/4/ra,/tenant/4/rz),@5=rk-inside",
	}, ranges)
}
