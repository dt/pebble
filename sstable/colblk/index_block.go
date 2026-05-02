// Copyright 2024 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package colblk

import (
	"bytes"
	"slices"
	"unsafe"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/internal/binfmt"
	"github.com/cockroachdb/pebble/internal/invariants"
	"github.com/cockroachdb/pebble/internal/treeprinter"
	"github.com/cockroachdb/pebble/internal/treesteps"
	"github.com/cockroachdb/pebble/sstable/block"
	"github.com/cockroachdb/pebble/sstable/blockiter"
)

const indexBlockCustomHeaderSize = 0

// IndexBlockWriter writes columnar index blocks. The writer is used for both
// first-level and second-level index blocks. The index block schema consists of
// three primary columns:
//   - Separators: a user key that is ≥ the largest user key in the
//     corresponding entry, and ≤ the smallest user key in the next entry.
//     Note that this allows consecutive separators to be equal. This is
//     possible when snapshots required we preserve duplicate user keys at
//     different sequence numbers.
//   - Offsets: the offset of the end of the corresponding block.
//   - Lengths: the length in bytes of the corresponding block.
//   - Block properties: a slice encoding arbitrary user-defined block
//     properties.
//
// TODO(jackson): Consider splitting separators into prefixes and suffixes (even
// without user-defined columns). This would allow us to use prefix compression
// for the prefix. Separators should typically be suffixless unless two KVs with
// the same prefix straddle a block boundary. We would need to use a buffer to
// materialize the separator key when we need to use it outside the context of
// seeking within the block.
type IndexBlockWriter struct {
	separators      RawBytesBuilder
	offsets         UintBuilder
	lengths         UintBuilder
	blockProperties RawBytesBuilder
	rows            int
	enc             BlockEncoder
}

const (
	indexBlockColumnSeparator = iota
	indexBlockColumnOffsets
	indexBlockColumnLengths
	indexBlockColumnBlockProperties
	indexBlockColumnCount
)

// Init initializes the index block writer.
func (w *IndexBlockWriter) Init() {
	w.separators.Init()
	w.offsets.Init()
	w.lengths.Init()
	w.blockProperties.Init()
	w.rows = 0
}

// Reset resets the index block writer to its initial state, retaining buffers.
func (w *IndexBlockWriter) Reset() {
	w.separators.Reset()
	w.offsets.Reset()
	w.lengths.Reset()
	w.blockProperties.Reset()
	w.rows = 0
	w.enc.Reset()
}

// Rows returns the number of entries in the index block so far.
func (w *IndexBlockWriter) Rows() int {
	return w.rows
}

// AddBlockHandle adds a new separator and end offset of a data block to the
// index block.  Add returns the index of the row.
//
// AddBlockHandle should only be used for first-level index blocks.
func (w *IndexBlockWriter) AddBlockHandle(
	separator []byte, handle block.Handle, blockProperties []byte,
) int {
	idx := w.rows
	w.separators.Put(separator)
	w.offsets.Set(w.rows, handle.Offset)
	w.lengths.Set(w.rows, handle.Length)
	w.blockProperties.Put(blockProperties)
	w.rows++
	return idx
}

// UnsafeSeparator returns the separator of the i'th entry.
func (w *IndexBlockWriter) UnsafeSeparator(i int) []byte {
	return w.separators.UnsafeGet(i)
}

// Size returns the size of the pending index block.
func (w *IndexBlockWriter) Size() int {
	return w.size(w.rows)
}

func (w *IndexBlockWriter) size(rows int) int {
	off := HeaderSize(indexBlockColumnCount, indexBlockCustomHeaderSize)
	off = w.separators.Size(rows, off)
	off = w.offsets.Size(rows, off)
	off = w.lengths.Size(rows, off)
	off = w.blockProperties.Size(rows, off)
	off++
	return int(off)
}

// Finish serializes the pending index block, including the first [rows] rows.
// The value of [rows] must be Rows() or Rows()-1.
func (w *IndexBlockWriter) Finish(rows int) []byte {
	if invariants.Enabled && rows != w.rows && rows != w.rows-1 {
		panic(errors.AssertionFailedf("index block has %d rows; asked to finish %d", errors.Safe(w.rows), errors.Safe(rows)))
	}

	w.enc.Init(w.size(rows), Header{
		Version: Version1,
		Columns: indexBlockColumnCount,
		Rows:    uint32(rows),
	}, indexBlockCustomHeaderSize)
	w.enc.Encode(rows, &w.separators)
	w.enc.Encode(rows, &w.offsets)
	w.enc.Encode(rows, &w.lengths)
	w.enc.Encode(rows, &w.blockProperties)
	return w.enc.Finish()
}

// An IndexBlockDecoder reads columnar index blocks.
type IndexBlockDecoder struct {
	separators RawBytes
	offsets    UnsafeUints
	lengths    UnsafeUints // only used for second-level index blocks
	blockProps RawBytes
	bd         BlockDecoder
}

// Init initializes the index block decoder with the given serialized index
// block.
func (r *IndexBlockDecoder) Init(data []byte) {
	r.bd.Init(data, indexBlockCustomHeaderSize)
	r.separators = r.bd.RawBytes(indexBlockColumnSeparator)
	r.offsets = r.bd.Uints(indexBlockColumnOffsets)
	r.lengths = r.bd.Uints(indexBlockColumnLengths)
	r.blockProps = r.bd.RawBytes(indexBlockColumnBlockProperties)
}

// DebugString prints a human-readable explanation of the keyspan block's binary
// representation.
func (r *IndexBlockDecoder) DebugString() string {
	f := binfmt.New(r.bd.data).LineWidth(20)
	tp := treeprinter.New()
	r.Describe(f, tp.Child("index-block-decoder"))
	return tp.String()
}

// Describe describes the binary format of the index block, assuming f.Offset()
// is positioned at the beginning of the same index block described by r.
func (r *IndexBlockDecoder) Describe(f *binfmt.Formatter, tp treeprinter.Node) {
	// Set the relative offset. When loaded into memory, the beginning of blocks
	// are aligned. Padding that ensures alignment is done relative to the
	// current offset. Setting the relative offset ensures that if we're
	// describing this block within a larger structure (eg, f.Offset()>0), we
	// compute padding appropriately assuming the current byte f.Offset() is
	// aligned.
	f.SetAnchorOffset()

	n := tp.Child("index block header")
	r.bd.HeaderToBinFormatter(f, n)
	for i := 0; i < indexBlockColumnCount; i++ {
		r.bd.ColumnToBinFormatter(f, n, i, int(r.bd.header.Rows))
	}
	f.HexBytesln(1, "block padding byte")
	f.ToTreePrinter(n)
}

// IndexIter is an iterator over the block entries in an index block.
type IndexIter struct {
	compare base.Compare
	split   base.Split
	d       *IndexBlockDecoder
	n       int
	row     int

	syntheticPrefixAndSuffix blockiter.SyntheticPrefixAndSuffix
	// blockPrefixSubstitution, if set, replaces a leading source prefix with a
	// destination prefix when emitting separators, and is inverted on incoming
	// seek keys. Mutually exclusive with a synthetic prefix.
	blockPrefixSubstitution blockiter.BlockPrefixSubstitution

	h block.BufferHandle
	// TODO(radu): remove allocDecoder and require any Init callers to provide the
	// decoder.
	allocDecoder IndexBlockDecoder
	keyBuf       []byte
	// seekKeyBuf is a scratch buffer used to translate seek keys from
	// destination-prefix space to storage-prefix space when a
	// BlockPrefixSubstitution is in effect. It is reused across Seek calls;
	// its contents are only valid for the duration of the underlying search.
	seekKeyBuf []byte
}

var _ blockiter.Index = (*IndexIter)(nil)

// InitWithDecoder initializes an index iterator from the provided decoder.
func (i *IndexIter) InitWithDecoder(
	comparer *base.Comparer, d *IndexBlockDecoder, transforms blockiter.Transforms,
) {
	i.compare = comparer.Compare
	i.split = comparer.Split
	i.d = d
	i.n = int(d.bd.header.Rows)
	i.row = -1
	i.syntheticPrefixAndSuffix = transforms.SyntheticPrefixAndSuffix
	i.blockPrefixSubstitution = transforms.BlockPrefixSubstitution
	// Leave h, allocDecoder, keyBuf, seekKeyBuf unchanged.
}

// Init initializes an iterator from the provided block data slice.
func (i *IndexIter) Init(
	comparer *base.Comparer, blk []byte, transforms blockiter.Transforms,
) error {
	i.h.Release()
	i.h = block.BufferHandle{}
	i.allocDecoder.Init(blk)
	i.InitWithDecoder(comparer, &i.allocDecoder, transforms)
	return nil
}

// InitHandle initializes an iterator from the provided block handle.
func (i *IndexIter) InitHandle(
	comparer *base.Comparer, blk block.BufferHandle, transforms blockiter.Transforms,
) error {
	i.h.Release()
	i.h = blk
	d := (*IndexBlockDecoder)(unsafe.Pointer(blk.BlockMetadata()))
	i.InitWithDecoder(comparer, d, transforms)
	return nil
}

// RowIndex returns the index of the block entry at the iterator's current
// position.
func (i *IndexIter) RowIndex() int {
	return i.row
}

// Valid returns true if the iterator is currently positioned at a valid block
// handle.
func (i *IndexIter) Valid() bool {
	return 0 <= i.row && i.row < i.n
}

// Invalidate invalidates the block iterator, removing references to the block
// it was initialized with.
func (i *IndexIter) Invalidate() {
	i.d = nil
	i.n = 0
}

// IsDataInvalidated returns true when the iterator has been invalidated
// using an Invalidate call. NB: this is different from Valid.
func (i *IndexIter) IsDataInvalidated() bool {
	return i.d == nil
}

// Handle returns the underlying block buffer handle, if the iterator was
// initialized with one.
func (i *IndexIter) Handle() block.BufferHandle {
	return i.h
}

// Separator returns the separator at the iterator's current position. The
// iterator must be positioned at a valid row.
func (i *IndexIter) Separator() []byte {
	key := i.d.separators.At(i.row)
	if i.syntheticPrefixAndSuffix.IsUnset() && !i.blockPrefixSubstitution.IsSet() {
		return key
	}
	return i.applyTransforms(key)
}

// SeparatorLT returns true if the separator at the iterator's current
// position is strictly less than the provided key.
func (i *IndexIter) SeparatorLT(key []byte) bool {
	return i.compare(i.Separator(), key) < 0
}

// SeparatorGT returns true if the separator at the iterator's current position
// is strictly greater than (or equal, if orEqual=true) the provided key.
func (i *IndexIter) SeparatorGT(key []byte, inclusively bool) bool {
	cmp := i.compare(i.Separator(), key)
	return cmp > 0 || (cmp == 0 && inclusively)
}

func (i *IndexIter) applyTransforms(key []byte) []byte {
	syntheticPrefix := i.syntheticPrefixAndSuffix.Prefix()
	syntheticSuffix := i.syntheticPrefixAndSuffix.Suffix()
	if syntheticSuffix.IsSet() {
		key = key[:i.split(key)]
	}
	// BlockPrefixSubstitution and SyntheticPrefix are mutually exclusive. If a
	// substitution is set, strip its Src from the stored separator and treat
	// Dst as the implicit prefix to prepend.
	var prefix []byte
	if sub := i.blockPrefixSubstitution; sub.IsSet() {
		// If the stored separator doesn't start with Src, the block to which
		// it points sits entirely outside the substitution's source range.
		// Project such separators to a sentinel that sorts (in dst space)
		// either before all dst-prefixed keys (if the storage key is < Src)
		// or after all dst-prefixed keys (if the storage key is > Src or its
		// immediate successor). This lets bounds checks stop iteration
		// cleanly without producing a garbled translation.
		//
		// Sentinels are appended with a trailing zero byte so that comparers
		// that interpret the trailing byte as a suffix length (e.g.
		// cockroachkvs.Compare uses key[len(key)-1] as the encoded suffix
		// length) treat the sentinel as a valid zero-suffix key. Without the
		// trailing zero, a sentinel like the bytes [0xfe 0x05] would have
		// its last byte interpreted as a 5-byte suffix length, producing a
		// negative slice index when Compare strips the suffix region. The
		// trailing zero is bytewise-stable for sort order and is valid input
		// for every Pebble comparer in tree.
		if !bytes.HasPrefix(key, sub.Src) {
			i.keyBuf = i.keyBuf[:0]
			if bytes.Compare(key, sub.Src) < 0 {
				// Pre-Src territory. Emit a sentinel that sorts before any
				// non-empty dst-prefixed key. The empty key sorts before all
				// non-empty keys for any reasonable comparer (and explicitly
				// for cockroachkvs.Compare, which length-compares zero-length
				// inputs). Lower-bound seek paths handle the pre-Src case
				// directly in SeekGE; this projection is for SeparatorLT-
				// style bounds comparisons reached via the iter forward
				// path.
				return i.keyBuf
			}
			// Post-Src territory. Synthesize a key strictly greater than any
			// dst-prefixed key by emitting Dst with the last byte
			// incremented (wrapping 0xff with an appended sentinel) and
			// then a trailing zero byte for cockroachkvs compatibility.
			i.keyBuf = append(i.keyBuf, sub.Dst...)
			n := len(i.keyBuf) - 1
			if i.keyBuf[n] != 0xff {
				i.keyBuf[n]++
			} else {
				i.keyBuf = append(i.keyBuf, 0xff)
			}
			i.keyBuf = append(i.keyBuf, 0x00)
			return i.keyBuf
		}
		key = key[len(sub.Src):]
		prefix = sub.Dst
	} else {
		prefix = syntheticPrefix
	}
	i.keyBuf = slices.Grow(i.keyBuf[:0], len(prefix)+len(key)+len(syntheticSuffix))
	i.keyBuf = append(i.keyBuf, prefix...)
	i.keyBuf = append(i.keyBuf, key...)
	i.keyBuf = append(i.keyBuf, syntheticSuffix...)
	return i.keyBuf
}

// BlockHandleWithProperties decodes the block handle with any encoded
// properties at the iterator's current position.
func (i *IndexIter) BlockHandleWithProperties() (block.HandleWithProperties, error) {
	if invariants.Enabled && !i.Valid() {
		panic(errors.AssertionFailedf("invalid row %d (n=%d)", errors.Safe(i.row), errors.Safe(i.n)))
	}
	return block.HandleWithProperties{
		Handle: block.Handle{
			Offset: i.d.offsets.At(i.row),
			Length: i.d.lengths.At(i.row),
		},
		Props: i.d.blockProps.At(i.row),
	}, nil
}

// SeekGE seeks the index iterator to the first block entry with a separator key
// greater or equal to the given key. It returns false if the seek key is
// greater than all index block separators.
func (i *IndexIter) SeekGE(key []byte) bool {
	// If a BlockPrefixSubstitution is in effect, the incoming seek key is in
	// destination-prefix space while the stored separators are in
	// storage-prefix space. Invert the seek key (strip Dst, prepend Src) so we
	// can compare directly against stored separators in the hot loop. If the
	// incoming key's leading bytes don't match Dst, route to the appropriate
	// extreme: a key that sorts before all dst-space keys lands at row 0; a
	// key that sorts after all dst-space keys lands past the end.
	if sub := i.blockPrefixSubstitution; sub.IsSet() {
		dstLen := len(sub.Dst)
		var keyPrefix []byte
		if len(key) <= dstLen {
			keyPrefix = key
			key = nil
		} else {
			keyPrefix = key[:dstLen]
			key = key[dstLen:]
		}
		if cmp := bytes.Compare(keyPrefix, sub.Dst); cmp != 0 {
			if cmp < 0 {
				i.row = 0
				return i.n > 0
			}
			i.row = i.n
			return false
		}
		// Translate to storage-prefix space by prepending Src.
		i.seekKeyBuf = append(i.seekKeyBuf[:0], sub.Src...)
		i.seekKeyBuf = append(i.seekKeyBuf, key...)
		key = i.seekKeyBuf
	}
	// Define f(-1) == false and f(upper) == true.
	// Invariant: f(index-1) == false, f(upper) == true.
	index, upper := 0, i.n
	for index < upper {
		h := int(uint(index+upper) >> 1) // avoid overflow when computing h
		// index ≤ h < upper

		// TODO(jackson): Is Bytes.At or Bytes.Slice(Bytes.Offset(h),
		// Bytes.Offset(h+1)) faster in this code?
		separator := i.d.separators.At(h)
		if i.syntheticPrefixAndSuffix.HasSuffix() {
			// We've inverted the seek key into storage-prefix space (if
			// applicable), so we only need to materialize the synthetic
			// suffix (which lives in suffix-space, independent of any
			// prefix substitution). The synthetic prefix path retains its
			// previous behavior since substitution and synthetic prefix are
			// mutually exclusive.
			// TODO(radu): compare without materializing the transformed key.
			syntheticPrefix := i.syntheticPrefixAndSuffix.Prefix()
			syntheticSuffix := i.syntheticPrefixAndSuffix.Suffix()
			sepKey := separator[:i.split(separator)]
			i.keyBuf = slices.Grow(i.keyBuf[:0], len(syntheticPrefix)+len(sepKey)+len(syntheticSuffix))
			i.keyBuf = append(i.keyBuf, syntheticPrefix...)
			i.keyBuf = append(i.keyBuf, sepKey...)
			i.keyBuf = append(i.keyBuf, syntheticSuffix...)
			separator = i.keyBuf
		} else if i.syntheticPrefixAndSuffix.HasPrefix() {
			// Pure synthetic prefix (no suffix and, by mutual exclusion, no
			// BlockPrefixSubstitution). Materialize the prefixed separator.
			// TODO(radu): compare without materializing the transformed key.
			separator = i.applyTransforms(separator)
		}
		// TODO(radu): experiment with splitting the separator prefix and suffix in
		// separate columns and using bytes.Compare() on the prefix in the hot path.
		c := i.compare(key, separator)
		if c > 0 {
			index = h + 1 // preserves f(index-1) == false
		} else {
			upper = h // preserves f(upper) == true
		}
	}
	// index == upper, f(index-1) == false, and f(upper) (= f(index)) == true  =>  answer is index.
	i.row = index
	return index < i.n
}

// First seeks index iterator to the first block entry. It returns false if the
// index block is empty.
func (i *IndexIter) First() bool {
	i.row = 0
	return i.n > 0
}

// Last seeks index iterator to the last block entry. It returns false if the
// index block is empty.
func (i *IndexIter) Last() bool {
	i.row = i.n - 1
	return i.n > 0
}

// Next steps the index iterator to the next block entry. It returns false if
// the index block is exhausted in the forward direction. A call to Next while
// already exhausted in the forward direction is a no-op.
//
// When a BlockPrefixSubstitution is configured, separators whose stored bytes
// don't start with sub.Src refer to blocks that lie outside the substitution
// range. In the forward direction we treat such separators as past-end
// (returning false) to avoid loading blocks whose keys cannot be safely
// translated to dst space.
//
// The exception is the LAST index entry. The colblk index writer emits the
// last entry's separator as the SST's overall upper bound — Successor of the
// largest key in that block — which for a key whose first byte is far below
// 0xff can shrink to a single high byte that does NOT itself start with
// sub.Src. The corresponding last data block IS in the substitution range,
// though, so we must not treat the last entry as past-end on the basis of
// its separator alone. For the last entry we instead consult the previous
// separator (the upper bound of the prior in-range block); if even that is
// past sub.Src then this last block is also out of range, otherwise the
// last block is in range and we keep it.
func (i *IndexIter) Next() bool {
	i.row = min(i.n, i.row+1)
	if i.row < i.n {
		if sub := i.blockPrefixSubstitution; sub.IsSet() {
			sep := i.d.separators.At(i.row)
			isLast := i.row == i.n-1
			if isLast && i.row > 0 {
				// Use the prior separator instead of the last-entry sentinel.
				sep = i.d.separators.At(i.row - 1)
			}
			if !bytes.HasPrefix(sep, sub.Src) && bytes.Compare(sep, sub.Src) > 0 {
				// This separator (and all subsequent ones, since separators
				// are sorted) lies past the substitution range. Treat as
				// exhausted.
				i.row = i.n
				return false
			}
		}
	}
	return i.row < i.n
}

// Prev steps the index iterator to the previous block entry. It returns false
// if the index block is exhausted in the reverse direction. A call to Prev
// while already exhausted in the reverse direction is a no-op.
//
// With a BlockPrefixSubstitution, separators whose stored bytes are < sub.Src
// refer to blocks whose UPPER bound lies entirely before the substitution
// range. We treat such separators as exhausted in the reverse direction.
func (i *IndexIter) Prev() bool {
	i.row = max(-1, i.row-1)
	if i.row >= 0 {
		if sub := i.blockPrefixSubstitution; sub.IsSet() {
			sep := i.d.separators.At(i.row)
			if !bytes.HasPrefix(sep, sub.Src) && bytes.Compare(sep, sub.Src) < 0 {
				i.row = -1
				return false
			}
		}
	}
	return i.row >= 0 && i.row < i.n
}

// Close closes the iterator, releasing any resources it holds.
func (i *IndexIter) Close() error {
	i.h.Release()
	i.h = block.BufferHandle{}
	i.d = nil
	i.n = 0
	i.syntheticPrefixAndSuffix = blockiter.SyntheticPrefixAndSuffix{}
	i.blockPrefixSubstitution = blockiter.BlockPrefixSubstitution{}
	return nil
}

func (i *IndexIter) TreeStepsNode() treesteps.NodeInfo {
	ni := treesteps.NodeInfof(i, "colblk.IndexIter")
	if i.Valid() {
		ni.AddPropf("at", "%s", i.Separator())
	} else {
		ni.AddPropf("not positioned", "")
	}
	return ni
}
