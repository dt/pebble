// Copyright 2026 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package pebble

import (
	"context"
	"fmt"
	"testing"

	"github.com/cockroachdb/crlib/testutils/leaktest"
	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/testkeys"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/stretchr/testify/require"
)

// openCloneTestDB opens an in-memory DB at the FormatPrefixSubstitution format
// major version (or the provided lower version, for the format-gate test).
func openCloneTestDB(t *testing.T, fmv FormatMajorVersion) *DB {
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

func TestVirtualClone_StraddlingSrcSpanRejected(t *testing.T) {
	defer leaktest.AfterTest(t)()
	d := openCloneTestDB(t, FormatPrefixSubstitution)
	defer func() { require.NoError(t, d.Close()) }()

	srcPrefix := []byte("/tenant/1/")
	dstPrefix := []byte("/tenant/4/")
	value := []byte("v")

	// Mix /tenant/0/ and /tenant/1/ keys in one SST so the resulting file
	// straddles the /tenant/1/ span.
	setMany(t, d, []string{
		"/tenant/0/key0",
		"/tenant/1/key1",
		"/tenant/1/key2",
	}, value)
	require.NoError(t, d.Flush())

	srcSpan := KeyRange{Start: srcPrefix, End: []byte("/tenant/2/")}
	err := d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix)
	require.Error(t, err)
	require.ErrorIs(t, err, ErrUnsupportedClone, "got %v", err)
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

func TestVirtualClone_DestinationConflict(t *testing.T) {
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
	err := d.VirtualClone(context.Background(), srcSpan, srcPrefix, dstPrefix)
	require.Error(t, err)
	require.ErrorIs(t, err, ErrUnsupportedClone, "got %v", err)
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
