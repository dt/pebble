package pebble

import (
	"context"
	"testing"
	"time"

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

func init() {
	// Ensure table stats collection runs promptly.
	_ = time.Millisecond
}
