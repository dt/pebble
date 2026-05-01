// Copyright 2025 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package blockiter

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTransforms(t *testing.T) {
	require.True(t, NoTransforms.NoTransforms())
	var transforms Transforms
	require.True(t, transforms.NoTransforms())
	require.True(t, transforms.SyntheticPrefixAndSuffix.IsUnset())
	transforms.SyntheticPrefixAndSuffix = MakeSyntheticPrefixAndSuffix([]byte{}, []byte{})
	require.True(t, transforms.SyntheticPrefixAndSuffix.IsUnset())
	require.True(t, transforms.NoTransforms())
	transforms.HideObsoletePoints = true
	require.False(t, transforms.NoTransforms())

	transforms = NoTransforms
	transforms.SyntheticSeqNum = 123
	require.False(t, transforms.NoTransforms())

	transforms = NoTransforms
	transforms.SyntheticPrefixAndSuffix = MakeSyntheticPrefixAndSuffix([]byte{1}, []byte{})
	require.False(t, transforms.NoTransforms())

	transforms = NoTransforms
	transforms.SyntheticPrefixAndSuffix = MakeSyntheticPrefixAndSuffix([]byte{}, []byte{1})
	require.False(t, transforms.NoTransforms())

	// A BlockPrefixSubstitution with non-empty Src disables the noTransforms
	// fast path. This locks in the invariant that callers using the
	// substitution cannot accidentally take the no-transform code path which
	// would emit untranslated keys.
	transforms = NoTransforms
	transforms.BlockPrefixSubstitution = BlockPrefixSubstitution{Src: []byte("/a/"), Dst: []byte("/b/")}
	require.True(t, transforms.BlockPrefixSubstitution.IsSet())
	require.False(t, transforms.NoTransforms())
}

func TestBlockPrefixSubstitution(t *testing.T) {
	// IsSet requires a non-empty Src; a non-empty Dst alone does not count.
	require.False(t, BlockPrefixSubstitution{}.IsSet())
	require.False(t, BlockPrefixSubstitution{Dst: []byte("/x/")}.IsSet())
	require.True(t, BlockPrefixSubstitution{Src: []byte("/x/")}.IsSet())
	require.True(t, BlockPrefixSubstitution{Src: []byte("/a/"), Dst: []byte("/b/")}.IsSet())

	// Round-trip Apply/Invert for a variety of key shapes.
	cases := []struct {
		name string
		src  []byte
		dst  []byte
		keys [][]byte
	}{
		{
			name: "equal-length",
			src:  []byte("/tenant/1/"),
			dst:  []byte("/tenant/4/"),
			keys: [][]byte{
				[]byte("/tenant/1/"), // exactly Src
				[]byte("/tenant/1/a"),
				[]byte("/tenant/1/abcdef"),
			},
		},
		{
			name: "longer-dst",
			src:  []byte("/t/1/"),
			dst:  []byte("/tenant/very-long/"),
			keys: [][]byte{
				[]byte("/t/1/"),
				[]byte("/t/1/x"),
			},
		},
		{
			name: "shorter-dst",
			src:  []byte("/tenant/12345/"),
			dst:  []byte("/t/9/"),
			keys: [][]byte{
				[]byte("/tenant/12345/"),
				[]byte("/tenant/12345/zzz"),
			},
		},
		{
			name: "empty-dst-pure-strip",
			src:  []byte("/tenant/1/"),
			dst:  nil,
			keys: [][]byte{
				[]byte("/tenant/1/"),
				[]byte("/tenant/1/foo"),
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := BlockPrefixSubstitution{Src: tc.src, Dst: tc.dst}
			for _, k := range tc.keys {
				applied := s.Apply(k)
				inverted := s.Invert(applied)
				require.Equal(t, string(k), string(inverted),
					"Invert(Apply(%q)) = %q, expected %q", k, inverted, k)
			}
		})
	}

	// Apply panics on a key that does not start with Src.
	s := BlockPrefixSubstitution{Src: []byte("/a/"), Dst: []byte("/b/")}
	require.Panics(t, func() { s.Apply([]byte("/x/foo")) })
	// Invert panics on a key that does not start with Dst.
	require.Panics(t, func() { s.Invert([]byte("/x/foo")) })
	// Apply panics if Src is empty (callers should use SyntheticPrefix).
	require.Panics(t, func() {
		(BlockPrefixSubstitution{Dst: []byte("/b/")}).Apply([]byte("foo"))
	})
}

func TestFragmentTransforms(t *testing.T) {
	require.True(t, NoFragmentTransforms.NoTransforms())
	var transforms FragmentTransforms
	require.True(t, transforms.NoTransforms())
	require.True(t, transforms.SyntheticPrefixAndSuffix.IsUnset())
	transforms.SyntheticPrefixAndSuffix = MakeSyntheticPrefixAndSuffix([]byte{}, []byte{})
	require.True(t, transforms.SyntheticPrefixAndSuffix.IsUnset())
	require.True(t, transforms.NoTransforms())

	transforms.SyntheticSeqNum = 123
	require.False(t, transforms.NoTransforms())

	transforms = NoFragmentTransforms
	transforms.SyntheticPrefixAndSuffix = MakeSyntheticPrefixAndSuffix([]byte{1}, []byte{})
	require.False(t, transforms.NoTransforms())

	transforms = NoFragmentTransforms
	transforms.SyntheticPrefixAndSuffix = MakeSyntheticPrefixAndSuffix([]byte{}, []byte{1})
	require.False(t, transforms.NoTransforms())
}

func TestSyntheticPrefixAndSuffix(t *testing.T) {
	var ps SyntheticPrefixAndSuffix

	require.True(t, ps.IsUnset())
	require.False(t, ps.HasPrefix())
	require.Nil(t, ps.Prefix())
	require.Zero(t, ps.PrefixLen())
	require.False(t, ps.HasSuffix())
	require.Zero(t, ps.SuffixLen())
	require.Nil(t, ps.Suffix())

	ps = MakeSyntheticPrefixAndSuffix([]byte("some-prefix"), []byte("suffix"))
	require.False(t, ps.IsUnset())
	require.True(t, ps.HasPrefix())
	require.Equal(t, uint32(11), ps.PrefixLen())
	require.Equal(t, "some-prefix", string(ps.Prefix()))
	require.True(t, ps.HasSuffix())
	require.Equal(t, uint32(6), ps.SuffixLen())
	require.Equal(t, "suffix", string(ps.Suffix()))

	ps = MakeSyntheticPrefixAndSuffix([]byte("some-prefix"), []byte{})
	require.False(t, ps.IsUnset())
	require.True(t, ps.HasPrefix())
	require.Equal(t, uint32(11), ps.PrefixLen())
	require.Equal(t, "some-prefix", string(ps.Prefix()))
	require.False(t, ps.HasSuffix())
	require.Zero(t, ps.SuffixLen())
	require.Nil(t, ps.Suffix())

	ps = MakeSyntheticPrefixAndSuffix([]byte("some-prefix"), []byte("suffix"))
	ps = ps.RemoveSuffix()
	require.False(t, ps.IsUnset())
	require.True(t, ps.HasPrefix())
	require.Equal(t, "some-prefix", string(ps.Prefix()))
	require.False(t, ps.HasSuffix())
	require.Zero(t, ps.SuffixLen())
	require.Nil(t, ps.Suffix())

	ps = MakeSyntheticPrefixAndSuffix([]byte{}, []byte("suffix"))
	require.False(t, ps.IsUnset())
	require.False(t, ps.HasPrefix())
	require.Zero(t, ps.PrefixLen())
	require.Nil(t, ps.Prefix())
	require.True(t, ps.HasSuffix())
	require.Equal(t, uint32(6), ps.SuffixLen())
	require.Equal(t, "suffix", string(ps.Suffix()))

	require.True(t, ps.RemoveSuffix().IsUnset())
}
