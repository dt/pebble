package pebble

import (
	"context"
	"testing"

	"github.com/cockroachdb/crlib/testutils/leaktest"
	"github.com/cockroachdb/pebble/cockroachkvs"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
)

// TestVirtualClone_SrcSideCompactionRace reproduces the "deleted table not in
// level" panic: a compaction with source-prefix inputs races with VirtualClone
// which replaces the source physical SST with a virtual stand-in. The cancel
// block in installClonePlan only checks dstSpan overlap, missing the srcSpan
// compaction.
func TestVirtualClone_SrcSideCompactionRace(t *testing.T) {
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

	cloneDone := make(chan error, 1)
	hookFired := false

	// Hook: after runCompaction finishes but before the compaction's
	// UpdateVersionLocked, d.mu is released and we inject a VirtualClone.
	// Clone install replaces the source physical (the compaction's input)
	// with a virtual stand-in. When the hook returns, the compaction
	// reacquires d.mu and tries to commit — if the cancel block missed
	// the srcSpan compaction, the commit panics.
	opts.private.testingCompactionBeforeApply = func() {
		if hookFired {
			return
		}
		hookFired = true
		cloneDone <- d_testSrcRace.VirtualClone(context.Background(),
			srcSpan_testSrcRace, srcPrefix_testSrcRace,
			dstSpan_testSrcRace, dstPrefix_testSrcRace)
	}

	d, err := Open("", opts)
	require.NoError(t, err)
	d_testSrcRace = d
	defer func() {
		d_testSrcRace = nil
		require.NoError(t, d.Close())
	}()

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

	srcPrefix_testSrcRace = []byte{0xfe, 0x8b}
	dstPrefix_testSrcRace = []byte{0xfe, 0x8c}
	srcSpan_testSrcRace = KeyRange{Start: enc(srcPrefix_testSrcRace), End: enc([]byte{0xfe, 0x8c})}
	dstSpan_testSrcRace = KeyRange{Start: enc(dstPrefix_testSrcRace), End: enc([]byte{0xfe, 0x8d})}

	// Write source data and flush to produce a physical SST in L0.
	for i := 0; i < 50; i++ {
		k := encMVCC(append(append([]byte{}, srcPrefix_testSrcRace...), byte(0x88), byte(i>>8), byte(i&0xff)), uint64(100+i))
		require.NoError(t, d.Set(k, []byte("v"), nil))
	}
	require.NoError(t, d.Flush())

	// Manual compaction over srcSpan. The hook fires between
	// runCompaction and UpdateVersionLocked, injecting VirtualClone.
	err = d.Compact(context.Background(), srcSpan_testSrcRace.Start, srcSpan_testSrcRace.End, false)

	// After fix: the compaction should be cancelled (not panic).
	// ErrCancelledCompaction is swallowed by Compact, so err == nil.
	require.NoError(t, err)

	// VirtualClone should have completed successfully.
	select {
	case cloneErr := <-cloneDone:
		require.NoError(t, cloneErr)
	default:
		t.Fatal("VirtualClone never ran")
	}
}

// Package-level vars to pass state into the testingCompactionBeforeApply hook
// (which is a plain func() with no args). Only used by
// TestVirtualClone_SrcSideCompactionRace.
var (
	d_testSrcRace          *DB
	srcSpan_testSrcRace    KeyRange
	dstSpan_testSrcRace    KeyRange
	srcPrefix_testSrcRace  []byte
	dstPrefix_testSrcRace  []byte
)
