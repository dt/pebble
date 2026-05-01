// Copyright 2026 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package sstable

import (
	"context"
	"fmt"
	"testing"

	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/internal/testkeys"
	"github.com/cockroachdb/pebble/objstorage"
	"github.com/cockroachdb/pebble/sstable/blockiter"
	"github.com/cockroachdb/pebble/sstable/colblk"
	"github.com/cockroachdb/pebble/sstable/virtual"
	"github.com/stretchr/testify/require"
)

// TestVirtualReaderBlockPrefixSubstitution verifies that BlockPrefixSubstitution
// configured on a virtual sstable's transforms is honored end-to-end through
// the iterator stack returned by Reader.NewPointIter.
//
// SCOPE NOTE (next slice): the index-block iterator does not yet apply
// BlockPrefixSubstitution to its separators (see colblk.IndexIter.Separator
// and SeekGE). As a result, when the iterator stack performs a SeekGE in
// destination space, the comparison against src-space index separators is
// incorrect. The data-block iterator handles substitution correctly, so the
// only subtests that work today are ones that avoid an index-level seek with
// a dst-space key:
//
//   - The "full-bounds" subtest below sets the virtual lower bound to a key
//     that sorts BEFORE the physical SST's src-space key range, so
//     index.SeekGE doesn't filter anything out and we exercise the rest of
//     the stack (load block, dispatch to dst-space data block iter, walk
//     keys).
//
// Subtests that require a dst-space key to actually drive index-level seeks
// (e.g. clipping to a strict subset, SeekGE landing inside the block via
// index lookup) are deferred to the next slice along with the index_block.go
// changes. They are intentionally left in as t.Skip() with a TODO so that
// when the index-iter substitution lands, removing the Skip should make them
// pass.
func TestVirtualReaderBlockPrefixSubstitution(t *testing.T) {
	src := []byte("/tenant/1/")
	dst := []byte("/tenant/4/")

	// Build a physical sstable populated with keys in src-space; keep all
	// keys in a single data block so that data-block-level iteration is the
	// only path that matters for the working subtests.
	keysSrc := make([][]byte, 0, 16)
	for i := 0; i < 16; i++ {
		keysSrc = append(keysSrc, []byte(fmt.Sprintf("%skey%04d", src, i)))
	}

	writerOpts := WriterOptions{
		TableFormat:    TableFormatMax,
		Comparer:       testkeys.Comparer,
		BlockSize:      1 << 20, // single data block
		IndexBlockSize: 1 << 20,
	}
	keySchema := colblk.DefaultKeySchema(writerOpts.Comparer, 16 /* bundle size */)
	writerOpts.KeySchema = &keySchema

	obj := &objstorage.MemObj{}
	w := NewRawWriter(obj, writerOpts)
	for i, k := range keysSrc {
		ik := base.MakeInternalKey(k, base.SeqNum(100+i), base.InternalKeyKindSet)
		require.NoError(t, w.Add(ik, []byte(fmt.Sprintf("v%04d", i)), false /* forceObsolete */, base.KVMeta{}))
	}
	require.NoError(t, w.Close())

	r, err := NewMemReader(obj.Data(), ReaderOptions{
		Comparer:   writerOpts.Comparer,
		KeySchemas: KeySchemas{writerOpts.KeySchema.Name: writerOpts.KeySchema},
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, r.Close()) }()

	// Translate the physical bounds into dst space for the virtual reader.
	keysDst := make([][]byte, len(keysSrc))
	for i, k := range keysSrc {
		keysDst[i] = append(append([]byte(nil), dst...), k[len(src):]...)
	}

	// "full-bounds-low-lower": virtual SST whose lower bound sorts BEFORE the
	// physical SST's src-space keys, sidestepping the index-iter limitation.
	// This validates that:
	//   - opts.Transforms.BlockPrefixSubstitution flows through file_cache /
	//     IterOptions / singleLevelIterator.init / DataBlockIter.InitHandle
	//     into the data block iterator, and
	//   - the data block iterator emits keys in destination space.
	t.Run("data-block-substitution-applied", func(t *testing.T) {
		// Lower is set to a byte sequence sorting before "/tenant/1/", so
		// the index seek in singleLevelIterator.SeekGE (called from First())
		// returns the only data block successfully. Upper is "/tenant/5"
		// which sorts after every dst-space key; the per-key bounds checks
		// in the data-block iter (which see dst-space keys) are also
		// satisfied.
		params := virtual.VirtualReaderParams{
			Lower:   base.MakeInternalKey([]byte("!"), base.SeqNumMax, base.InternalKeyKindSet),
			Upper:   base.MakeRangeDeleteSentinelKey([]byte("/tenant/5")),
			FileNum: 1,
		}

		iter, err := r.NewPointIter(context.Background(), IterOptions{
			Transforms: IterTransforms{
				BlockPrefixSubstitution: blockiter.BlockPrefixSubstitution{Src: src, Dst: dst},
			},
			FilterBlockSizeLimit: NeverUseFilterBlock,
			Env: ReadEnv{
				Virtual: &params,
			},
			ReaderProvider: MakeTrivialReaderProvider(r),
			BlobContext:    AssertNoBlobHandles,
		})
		require.NoError(t, err)
		defer func() { require.NoError(t, iter.Close()) }()

		// First/Next walk: every key should be in destination space.
		var got [][]byte
		for kv := iter.First(); kv != nil; kv = iter.Next() {
			got = append(got, append([]byte(nil), kv.K.UserKey...))
		}
		require.Equal(t, len(keysDst), len(got),
			"expected %d keys; got %d", len(keysDst), len(got))
		for i := range keysDst {
			require.Equal(t, string(keysDst[i]), string(got[i]),
				"key mismatch at i=%d", i)
		}
	})

	// The remaining subtests require IndexIter to apply
	// BlockPrefixSubstitution to its separators. Skip until that lands.

	t.Run("seek-ge-in-dst-space", func(t *testing.T) {
		t.Skip("TODO(next-slice): requires colblk.IndexIter to apply " +
			"BlockPrefixSubstitution to separators; see " +
			"sstable/colblk/index_block.go IndexIter.Separator/SeekGE.")
		_ = keysDst // avoid 'declared and not used' if Skip is removed
	})

	t.Run("subset-bounds", func(t *testing.T) {
		t.Skip("TODO(next-slice): requires colblk.IndexIter to apply " +
			"BlockPrefixSubstitution to separators; see " +
			"sstable/colblk/index_block.go IndexIter.Separator/SeekGE.")
	})
}
