package pebble

import (
	"context"
	"testing"

	"github.com/cockroachdb/crlib/testutils/leaktest"
	"github.com/cockroachdb/pebble/cockroachkvs"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
)

// TestVirtualClone_SyntheticSeqNumBoundsConsistency verifies that after
// installClonePlan bumps meta.SeqNums to activate SyntheticSeqNum, the
// table metadata's PointKeyBounds are consistent with the synthetic
// seqnum. Without this consistency, loadTableRangeDelStats' assertIter
// panics because the bounds' trailer seqnum (original) doesn't match
// the keyspan iter's rewritten trailer seqnum (synthetic/bumped).
func TestVirtualClone_SyntheticSeqNumBoundsConsistency(t *testing.T) {
	defer leaktest.AfterTest(t)()

	mem := vfs.NewMem()
	opts := &Options{
		Comparer:                    &cockroachkvs.Comparer,
		BlockPropertyCollectors:     cockroachkvs.BlockPropertyCollectors,
		FormatMajorVersion:          FormatPrefixSubstitution,
		FS:                          mem,
		KeySchema:                   cockroachkvs.KeySchema.Name,
		KeySchemas:                  sstable.MakeKeySchemas(&cockroachkvs.KeySchema),
		DisableAutomaticCompactions: true,
	}
	d, err := Open("", opts)
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

	// Flush 1: RANGEDEL only (low seqnum).
	require.NoError(t, d.DeleteRange(
		enc(srcPrefix), enc([]byte{0xfe, 0x8c}), nil))
	require.NoError(t, d.Flush())

	// Flush 2: point keys only (higher seqnum, overriding the rangedel).
	for i := 0; i < 10; i++ {
		k := encMVCC(append(append([]byte{}, srcPrefix...), byte(0x88), byte(i)), uint64(100+i))
		require.NoError(t, d.Set(k, []byte("v"), nil))
	}
	require.NoError(t, d.Flush())

	// Advance the global seqnum so exciseSeqNum is much higher than
	// the source's seqnums. The gap makes the assertIter violation
	// obvious: the bounds' trailer has the original seqnum (~2) while
	// the rewritten rangedel has exciseSeqNum (~112).
	for i := 0; i < 100; i++ {
		k := encMVCC([]byte{0x01, byte(i)}, uint64(1000+i))
		require.NoError(t, d.Set(k, []byte("padding"), nil))
	}
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: enc(srcPrefix), End: enc([]byte{0xfe, 0x8c})}
	dstSpan := KeyRange{Start: enc(dstPrefix), End: enc([]byte{0xfe, 0x8d})}
	require.NoError(t, d.VirtualClone(context.Background(),
		srcSpan, srcPrefix, dstSpan, dstPrefix))

	// Verify directly that for each cloned virtual SST with range
	// deletions, the PointKeyBounds trailers are consistent with the
	// SyntheticSeqNum. This is the invariant that loadTableRangeDelStats'
	// assertIter checks asynchronously.
	d.mu.Lock()
	v := d.mu.versions.currentVersion()
	v.Ref()
	d.mu.Unlock()
	defer v.Unref()
	for level := range v.Levels {
		for m := range v.Levels[level].All() {
			if !m.Virtual || !m.BlockPrefixSubstitution.IsSet() {
				continue
			}
			synSeqNum := m.SyntheticSeqNum()
			if synSeqNum == 0 {
				continue
			}
			smallest := m.PointKeyBounds.Smallest()
			t.Logf("L%d table %s: Virtual=%v SyntheticSeqNum=%d Smallest=%s (trailer=%d)",
				level, m.TableNum, m.Virtual, synSeqNum,
				smallest, smallest.Trailer)
			// The assertIter checks span.SmallestKey().Trailer > lower.Trailer.
			// With SyntheticSeqNum active, the keyspan iter rewrites all
			// trailers to synSeqNum. The bounds must also use synSeqNum,
			// otherwise the assert fires.
			expectedTrailer := base.MakeTrailer(base.SeqNum(synSeqNum), smallest.Trailer.Kind())
			if smallest.Trailer != expectedTrailer {
				t.Errorf("L%d table %s: PointKeyBounds.Smallest trailer=%d but SyntheticSeqNum=%d; "+
					"expected trailer=%d. This will cause assertIter panic in loadTableRangeDelStats.",
					level, m.TableNum, smallest.Trailer, synSeqNum, expectedTrailer)
			}
		}
	}

	// If we get here without a panic, the bounds are consistent.
	// Also verify we can read the cloned data.
	iter, err := d.NewIter(&IterOptions{
		LowerBound: dstSpan.Start,
		UpperBound: dstSpan.End,
	})
	require.NoError(t, err)
	count := 0
	for iter.First(); iter.Valid(); iter.Next() {
		count++
	}
	require.NoError(t, iter.Close())
	require.Greater(t, count, 0, "expected cloned data in dst span")
}

// TestVirtualClone_SyntheticSeqNumRangeDelAtSmallest verifies that
// table-stats collection doesn't panic when a cloned virtual SST has
// both a SET and a RANGEDEL starting at the same user key. After
// SyntheticSeqNum equalizes their seqnums, the RANGEDEL (kind 15)
// sorts before SET (kind 1) at the same seqnum, violating a
// SET-based assertIter lower bound.
func TestVirtualClone_SyntheticSeqNumRangeDelAtSmallest(t *testing.T) {
	defer leaktest.AfterTest(t)()

	mem := vfs.NewMem()
	opts := &Options{
		Comparer:                    &cockroachkvs.Comparer,
		BlockPropertyCollectors:     cockroachkvs.BlockPropertyCollectors,
		FormatMajorVersion:          FormatPrefixSubstitution,
		FS:                          mem,
		KeySchema:                   cockroachkvs.KeySchema.Name,
		KeySchemas:                  sstable.MakeKeySchemas(&cockroachkvs.KeySchema),
		DisableAutomaticCompactions: true,
	}
	d, err := Open("", opts)
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

	startKey := append(append([]byte{}, srcPrefix...), 0x88, 0x00)

	// RANGEDEL first (lower seqnum), starting at startKey.
	require.NoError(t, d.DeleteRange(
		enc(startKey), enc(append(append([]byte{}, srcPrefix...), 0x88, 0x10)), nil))
	// SET at the same user key (higher seqnum). In the source, SET
	// sorts first (higher seqnum = smaller internal key). After
	// SyntheticSeqNum, RANGEDEL sorts first (higher kind at same seqnum).
	require.NoError(t, d.Set(enc(startKey), []byte("v"), nil))
	for i := 1; i < 5; i++ {
		k := encMVCC(append(append([]byte{}, srcPrefix...), byte(0x88), byte(i)), uint64(100+i))
		require.NoError(t, d.Set(k, []byte("v"), nil))
	}
	require.NoError(t, d.Flush())

	// Advance seqnum.
	for i := 0; i < 100; i++ {
		k := encMVCC([]byte{0x01, byte(i)}, uint64(1000+i))
		require.NoError(t, d.Set(k, []byte("padding"), nil))
	}
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: enc(srcPrefix), End: enc([]byte{0xfe, 0x8c})}
	dstSpan := KeyRange{Start: enc(dstPrefix), End: enc([]byte{0xfe, 0x8d})}
	require.NoError(t, d.VirtualClone(context.Background(),
		srcSpan, srcPrefix, dstSpan, dstPrefix))

	// Force table-stats collection. The assertIter in
	// loadTableRangeDelStats panics without the fix because the
	// RANGEDEL at exciseSeqNum sorts before the SET-based lower bound.
	for d.collectTableStats() {
	}
	d.mu.Lock()
	for d.mu.tableStats.loading || len(d.mu.tableStats.pending) > 0 {
		d.mu.tableStats.cond.Wait()
	}
	d.mu.Unlock()
}

// TestVirtualClone_SyntheticSeqNumOverallBounds verifies that after the
// SyntheticSeqNum trailer rewrite, the overall bounds (Smallest/Largest)
// are consistent with the rewritten PointKeyBounds and RangeKeyBounds.
// When both point and range keys exist at the same user key, the trailer
// rewrite can change which bound type provides the overall smallest.
func TestVirtualClone_SyntheticSeqNumOverallBounds(t *testing.T) {
	defer leaktest.AfterTest(t)()

	mem := vfs.NewMem()
	opts := &Options{
		Comparer:                    &cockroachkvs.Comparer,
		BlockPropertyCollectors:     cockroachkvs.BlockPropertyCollectors,
		FormatMajorVersion:          FormatPrefixSubstitution,
		FS:                          mem,
		KeySchema:                   cockroachkvs.KeySchema.Name,
		KeySchemas:                  sstable.MakeKeySchemas(&cockroachkvs.KeySchema),
		DisableAutomaticCompactions: true,
	}
	d, err := Open("", opts)
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

	// RANGEKEYSET fully inside srcSpan (lower seqnum). Use a narrow
	// range that doesn't touch srcSpan boundaries, so it stays in the
	// cloned virtual (no forceL0 split).
	rangeKeyStart := append(append([]byte{}, srcPrefix...), 0x88, 0x00)
	rangeKeyEnd := append(append([]byte{}, srcPrefix...), 0x88, 0x10)
	require.NoError(t, d.RangeKeySet(
		enc(rangeKeyStart), enc(rangeKeyEnd), nil, []byte("val"), nil))
	// Point key at same user key as RANGEKEYSET start (higher seqnum).
	require.NoError(t, d.Set(enc(rangeKeyStart), []byte("v"), nil))
	for i := 0; i < 5; i++ {
		k := encMVCC(append(append([]byte{}, srcPrefix...), byte(0x88), byte(i)), uint64(100+i))
		require.NoError(t, d.Set(k, []byte("v"), nil))
	}
	require.NoError(t, d.Flush())

	// Advance the global seqnum so exciseSeqNum >> source seqnums.
	for i := 0; i < 100; i++ {
		k := encMVCC([]byte{0x01, byte(i)}, uint64(1000+i))
		require.NoError(t, d.Set(k, []byte("padding"), nil))
	}
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: enc(srcPrefix), End: enc([]byte{0xfe, 0x8c})}
	dstSpan := KeyRange{Start: enc(dstPrefix), End: enc([]byte{0xfe, 0x8d})}
	require.NoError(t, d.VirtualClone(context.Background(),
		srcSpan, srcPrefix, dstSpan, dstPrefix))

	// Validate all cloned virtuals. The "inconsistent range key bounds
	// relative to overall bounds" error fires if the trailer rewrite
	// doesn't update the overall bound types.
	d.mu.Lock()
	v := d.mu.versions.currentVersion()
	v.Ref()
	d.mu.Unlock()
	defer v.Unref()
	for level := range v.Levels {
		for m := range v.Levels[level].All() {
			t.Logf("L%d table %s: HasPointKeys=%v HasRangeKeys=%v Smallest=%s Largest=%s",
				level, m.TableNum, m.HasPointKeys, m.HasRangeKeys,
				m.Smallest(), m.Largest())
			if m.HasRangeKeys {
				t.Logf("  RangeKeyBounds: [%s - %s]",
					m.RangeKeyBounds.Smallest(), m.RangeKeyBounds.Largest())
			}
			if err := m.Validate(d.cmp, d.opts.Comparer.FormatKey); err != nil {
				t.Fatalf("L%d table %s: %v", level, m.TableNum, err)
			}
		}
	}
}
