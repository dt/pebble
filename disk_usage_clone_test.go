package pebble

import (
	"context"
	"testing"

	"github.com/cockroachdb/crlib/testutils/leaktest"
	"github.com/cockroachdb/pebble/internal/testkeys"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
)

func TestExclusiveBytesInSpan(t *testing.T) {
	defer leaktest.AfterTest(t)()

	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { validateAndCloseCloneTestDB(t, d) }()

	srcPrefix := []byte("aa")
	dstPrefix := []byte("cc")
	val := make([]byte, 100)

	// Write data into srcPrefix and flush.
	for i := 0; i < 20; i++ {
		k := []byte("aa" + string(rune('a'+i)))
		require.NoError(t, d.Set(k, val, nil))
	}
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("ab")}
	dstSpan := dstSpanForTest(srcSpan, srcPrefix, dstPrefix)

	// Before clone: srcSpan has a physical SST with exclusive bytes.
	preClone, err := d.ExclusiveBytesInSpan(srcSpan.Start, srcSpan.End)
	require.NoError(t, err)
	require.Greater(t, preClone, uint64(0), "srcSpan should have exclusive bytes before clone")

	// Clone src → dst.
	require.NoError(t, d.VirtualClone(context.Background(),
		srcSpan, srcPrefix, dstSpan, dstPrefix))

	// After clone: the source physical was replaced by a virtual stand-in.
	// The backing is now shared between the stand-in (srcSpan) and the
	// cloned virtual (dstSpan). useCount=2 → neither span owns it
	// exclusively.
	srcExclusive, err := d.ExclusiveBytesInSpan(srcSpan.Start, srcSpan.End)
	require.NoError(t, err)
	dstExclusive, err := d.ExclusiveBytesInSpan(dstSpan.Start, dstSpan.End)
	require.NoError(t, err)
	t.Logf("srcExclusive=%d dstExclusive=%d preClone=%d", srcExclusive, dstExclusive, preClone)

	// Both should be zero: the backing is shared (refcount 2).
	require.Equal(t, uint64(0), srcExclusive,
		"srcSpan should have 0 exclusive bytes (backing shared with dst)")
	require.Equal(t, uint64(0), dstExclusive,
		"dstSpan should have 0 exclusive bytes (backing shared with src)")

	// Write new data into srcSpan and compact to force the stand-in to
	// be rewritten as a new physical. This drops the src-side reference
	// to the shared backing, leaving the dst virtual as the sole user.
	for i := 0; i < 5; i++ {
		k := []byte("aa" + string(rune('a'+i)))
		require.NoError(t, d.Set(k, val, nil))
	}
	require.NoError(t, d.Flush())
	require.NoError(t, d.Compact(context.Background(),
		srcSpan.Start, srcSpan.End, false))

	srcAfter, err := d.ExclusiveBytesInSpan(srcSpan.Start, srcSpan.End)
	require.NoError(t, err)
	dstAfter, err := d.ExclusiveBytesInSpan(dstSpan.Start, dstSpan.End)
	require.NoError(t, err)
	t.Logf("srcAfter=%d dstAfter=%d", srcAfter, dstAfter)

	// srcSpan: the stand-in was rewritten → new physical → exclusive.
	require.Greater(t, srcAfter, uint64(0),
		"srcSpan should have exclusive bytes (new physical from compaction)")
	// dstSpan: the cloned virtual is now the sole user of the old backing
	// (refcount=1) → exclusive.
	require.Greater(t, dstAfter, uint64(0),
		"dstSpan should have exclusive bytes (sole user of old backing)")
}

func TestExclusiveBytesInSpan_Empty(t *testing.T) {
	defer leaktest.AfterTest(t)()

	mem := vfs.NewMem()
	d, err := Open("", &Options{
		Comparer:           testkeys.Comparer,
		FS:                 mem,
		FormatMajorVersion: FormatPrefixSubstitution,
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, d.Close()) }()

	bytes, err := d.ExclusiveBytesInSpan([]byte("a"), []byte("z"))
	require.NoError(t, err)
	require.Equal(t, uint64(0), bytes)
}
