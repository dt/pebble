package pebble

import (
	"context"
	"testing"

	"github.com/cockroachdb/crlib/testutils/leaktest"
	"github.com/cockroachdb/pebble/internal/testkeys"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
)

// TestVirtualClone_FlushBetweenSnapshotAndPrepare reproduces the "missing
// recent writes" bug: a background flush between phase A (buildClonePlan's
// version snapshot) and phase B (installClonePlanViaCommitPipeline's prepare)
// moves data from the memtable to L0. Phase B's prepare sees no memtable
// overlap (data already flushed), so it doesn't abort. But phase A's plan
// was built from the pre-flush version and misses the L0 file. The cloned
// virtual SSTs don't include the flushed data.
func TestVirtualClone_FlushBetweenSnapshotAndPrepare(t *testing.T) {
	defer leaktest.AfterTest(t)()

	mem := vfs.NewMem()
	opts := &Options{
		Comparer:                    testkeys.Comparer,
		FS:                          mem,
		FormatMajorVersion:          FormatPrefixSubstitution,
		DisableAutomaticCompactions: true,
	}

	var attempt int
	// Hook fires after phase A snapshots the version but before phase B.
	// On the first attempt, flush the memtable (moving data to L0 without
	// going through the abort-on-flush path). This simulates a background
	// flush that races with VirtualClone.
	opts.private.testingCloneAfterSnapshot = func(a int) {
		attempt = a
	}

	d, err := Open("", opts)
	require.NoError(t, err)
	defer func() { validateAndCloseCloneTestDB(t, d) }()

	srcPrefix := []byte("aa")
	dstPrefix := []byte("cc")

	// Write keys to srcSpan and leave them in the memtable.
	keys := []string{"aaa", "aab", "aac", "aad", "aae"}
	val := []byte("value")
	setMany(t, d, keys, val)

	// Install the hook AFTER writing so the writes are in the memtable.
	flushed := false
	d.opts.private.testingCloneAfterSnapshot = func(a int) {
		attempt = a
		if !flushed {
			// Flush the memtable. This moves the data from memtable to
			// L0, but phase A's snapshot is already taken (stale).
			flushed = true
			require.NoError(t, d.Flush())
		}
	}

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("ab")}
	dstSpan := dstSpanForTest(srcSpan, srcPrefix, dstPrefix)
	require.NoError(t, d.VirtualClone(context.Background(),
		srcSpan, srcPrefix, dstSpan, dstPrefix))

	// Verify ALL keys appear in dstSpan. Without the fix, the flushed
	// data is missing because phase A's snapshot predates the flush.
	dstKeys := scanRange(t, d, dstSpan.Start, dstSpan.End)
	require.Len(t, dstKeys, len(keys),
		"expected all %d keys in dstSpan; got %d (missing data from flush between snapshot and prepare)",
		len(keys), len(dstKeys))

	// The fix should have caused at least one abort+retry.
	require.Greater(t, attempt, 0,
		"expected at least one retry due to version change detection")
}
