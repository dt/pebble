// Copyright 2024 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package colblk

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"sync"
	"unsafe"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/internal/binfmt"
	"github.com/cockroachdb/pebble/internal/invariants"
	"github.com/cockroachdb/pebble/internal/keyspan"
	"github.com/cockroachdb/pebble/internal/treeprinter"
	"github.com/cockroachdb/pebble/internal/treesteps"
	"github.com/cockroachdb/pebble/sstable/block"
	"github.com/cockroachdb/pebble/sstable/blockiter"
)

// the keyspan header encodes a 32-bit count of the number of unique boundary
// user keys in the block.
const keyspanHeaderSize = 4

// keyspan block column indexes
const (
	// Columns with 1 row per unique boundary user key contained within the
	// block (with the count indicated via the keyspan custom block header).
	keyspanColBoundaryUserKeys   = 0
	keyspanColBoundaryKeyIndices = 1
	// Columns with 1 row per keyspan.Key (with the count indicated via the
	// columnar header's row count).
	keyspanColTrailers = 2
	keyspanColSuffixes = 3
	keyspanColValues   = 4
	keyspanColumnCount = 5
)

// A KeyspanBlockWriter writes keyspan blocks. See the colblk package
// documentation for more details on the schema.
type KeyspanBlockWriter struct {
	equal base.Equal

	// boundary columns
	boundaryUserKeys   RawBytesBuilder
	boundaryKeyIndexes UintBuilder

	// keyspan.Key columns
	trailers UintBuilder
	suffixes RawBytesBuilder
	values   RawBytesBuilder

	enc               BlockEncoder
	keyCount          int
	unsafeLastUserKey []byte
}

// Init initializes a keyspan block writer.
func (w *KeyspanBlockWriter) Init(equal base.Equal) {
	w.equal = equal
	w.boundaryUserKeys.Init()
	w.boundaryKeyIndexes.Init()
	w.trailers.Init()
	w.suffixes.Init()
	w.values.Init()
	w.keyCount = 0
	w.unsafeLastUserKey = nil
}

// Reset resets the keyspan block writer to an empty state, retaining memory for
// reuse.
func (w *KeyspanBlockWriter) Reset() {
	w.boundaryUserKeys.Reset()
	w.boundaryKeyIndexes.Reset()
	w.trailers.Reset()
	w.suffixes.Reset()
	w.values.Reset()
	w.enc.Reset()
	w.keyCount = 0
	w.unsafeLastUserKey = nil
}

// AddSpan appends a new Span to the pending block. Spans must already be
// fragmented (non-overlapping) and added in sorted order.
func (w *KeyspanBlockWriter) AddSpan(s keyspan.Span) {
	// When keyspans are fragmented, abutting spans share a user key. One span's
	// end key is the next span's start key.  Check if the previous user key
	// equals this span's start key, and avoid encoding it again if so.
	if w.unsafeLastUserKey == nil || !w.equal(w.unsafeLastUserKey, s.Start) {
		w.boundaryKeyIndexes.Set(w.boundaryUserKeys.rows, uint64(w.keyCount))
		w.boundaryUserKeys.Put(s.Start)
	}
	// The end key must be strictly greater than the start key and spans are
	// already sorted, so the end key is guaranteed to not be present in the
	// column yet. We need to encode it.
	w.boundaryKeyIndexes.Set(w.boundaryUserKeys.rows, uint64(w.keyCount+len(s.Keys)))
	w.boundaryUserKeys.Put(s.End)

	// Hold on to a slice of the copy of s.End we just added to the bytes
	// builder so that we can compare it to the next span's start key.
	w.unsafeLastUserKey = w.boundaryUserKeys.data[len(w.boundaryUserKeys.data)-len(s.End):]

	// Encode each keyspan.Key in the span.
	for i := range s.Keys {
		w.trailers.Set(w.keyCount, uint64(s.Keys[i].Trailer))
		w.suffixes.Put(s.Keys[i].Suffix)
		w.values.Put(s.Keys[i].Value)
		w.keyCount++
	}
}

// KeyCount returns the count of keyspan.Keys written to the writer.
func (w *KeyspanBlockWriter) KeyCount() int {
	return w.keyCount
}

// UnsafeBoundaryKeys returns the smallest and largest keys written to the
// keyspan block so far. The returned internal keys have user keys that point
// directly into the block writer's memory and must not be mutated.
func (w *KeyspanBlockWriter) UnsafeBoundaryKeys() (smallest, largest base.InternalKey) {
	if w.keyCount == 0 {
		return smallest, largest
	}
	smallest.UserKey = w.boundaryUserKeys.UnsafeGet(0)
	smallest.Trailer = base.InternalKeyTrailer(w.trailers.Get(0))
	largest.UserKey = w.boundaryUserKeys.UnsafeGet(w.boundaryUserKeys.rows - 1)
	largest.Trailer = base.MakeTrailer(base.SeqNumMax,
		base.InternalKeyTrailer(w.trailers.Get(w.keyCount-1)).Kind())
	return smallest, largest
}

// UnsafeLastSpan returns the start and end user keys of the last span written
// to the block and the trailer of its largest key. The returned keys point
// directly into the block writer's memory and must not be mutated.
func (w *KeyspanBlockWriter) UnsafeLastSpan() (
	start, end []byte,
	largestTrailer base.InternalKeyTrailer,
) {
	if w.keyCount == 0 {
		return nil, nil, 0
	}
	return w.boundaryUserKeys.UnsafeGet(w.boundaryUserKeys.rows - 2),
		w.boundaryUserKeys.UnsafeGet(w.boundaryUserKeys.rows - 1),
		base.InternalKeyTrailer(w.trailers.Get(w.keyCount - 1))
}

// Size returns the size of the pending block.
func (w *KeyspanBlockWriter) Size() int {
	off := HeaderSize(keyspanColumnCount, keyspanHeaderSize)
	// Span boundary columns (with userKeyCount elements).
	off = w.boundaryUserKeys.Size(w.boundaryUserKeys.rows, off)
	off = w.boundaryKeyIndexes.Size(w.boundaryUserKeys.rows, off)

	// keyspan.Key columns (with keyCount elements).
	off = w.trailers.Size(w.keyCount, off)
	off = w.suffixes.Size(w.keyCount, off)
	off = w.values.Size(w.keyCount, off)
	off++ // trailing padding
	return int(off)
}

// Finish finalizes the pending block and returns the encoded block.
func (w *KeyspanBlockWriter) Finish() []byte {
	w.enc.Init(w.Size(), Header{
		Version: Version1,
		Columns: keyspanColumnCount,
		Rows:    uint32(w.keyCount),
	}, keyspanHeaderSize)

	// The keyspan block has a 4-byte custom header used to encode the number of
	// user keys encoded within the user key and start indices columns. All
	// other columns have the number of rows indicated by the shared columnar
	// block header.
	binary.LittleEndian.PutUint32(w.enc.Data()[:keyspanHeaderSize], uint32(w.boundaryUserKeys.rows))

	// Columns with userKeyCount elements.
	w.enc.Encode(w.boundaryUserKeys.rows, &w.boundaryUserKeys)
	w.enc.Encode(w.boundaryUserKeys.rows, &w.boundaryKeyIndexes)
	// Columns with keyCount elements.
	w.enc.Encode(w.keyCount, &w.trailers)
	w.enc.Encode(w.keyCount, &w.suffixes)
	w.enc.Encode(w.keyCount, &w.values)
	return w.enc.Finish()
}

// String returns a string representation of the pending block's state.
func (w *KeyspanBlockWriter) String() string {
	var buf bytes.Buffer
	size := uint32(w.Size())
	fmt.Fprintf(&buf, "size=%d:\n", size)

	fmt.Fprint(&buf, "0: user keys:      ")
	w.boundaryUserKeys.WriteDebug(&buf, w.boundaryUserKeys.rows)
	fmt.Fprintln(&buf)
	fmt.Fprint(&buf, "1: start indices:  ")
	w.boundaryKeyIndexes.WriteDebug(&buf, w.boundaryUserKeys.rows)
	fmt.Fprintln(&buf)

	fmt.Fprint(&buf, "2: trailers:       ")
	w.trailers.WriteDebug(&buf, w.keyCount)
	fmt.Fprintln(&buf)
	fmt.Fprint(&buf, "3: suffixes:       ")
	w.suffixes.WriteDebug(&buf, w.keyCount)
	fmt.Fprintln(&buf)
	fmt.Fprint(&buf, "4: values:         ")
	w.values.WriteDebug(&buf, w.keyCount)
	fmt.Fprintln(&buf)

	return buf.String()
}

// A KeyspanDecoder exposes facilities for decoding a keyspan block. A
// KeyspanDecoder is safe for concurrent use after initialization.
type KeyspanDecoder struct {
	blockDecoder BlockDecoder
	// Span boundary columns with boundaryKeysCount elements.
	boundaryKeysCount  uint32
	boundaryKeys       RawBytes
	boundaryKeyIndices UnsafeUints

	// keyspan.Key columns with blockDecoder.header.Rows elements.
	trailers UnsafeUints
	suffixes RawBytes
	values   RawBytes
}

// Init initializes the keyspan decoder with the given block data.
func (d *KeyspanDecoder) Init(data []byte) {
	d.boundaryKeysCount = binary.LittleEndian.Uint32(data[:4])
	d.blockDecoder.Init(data, keyspanHeaderSize)
	// The boundary key columns have a different number of rows than the other
	// columns, so we call DecodeColumn directly, taking care to pass in
	// rows=r.boundaryKeysCount.
	d.boundaryKeys = DecodeColumn(&d.blockDecoder, keyspanColBoundaryUserKeys,
		d.boundaryKeysCount, DataTypeBytes, DecodeRawBytes)
	d.boundaryKeyIndices = DecodeColumn(&d.blockDecoder, keyspanColBoundaryKeyIndices,
		d.boundaryKeysCount, DataTypeUint, DecodeUnsafeUints)

	d.trailers = d.blockDecoder.Uints(keyspanColTrailers)
	d.suffixes = d.blockDecoder.RawBytes(keyspanColSuffixes)
	d.values = d.blockDecoder.RawBytes(keyspanColValues)
}

// DebugString prints a human-readable explanation of the keyspan block's binary
// representation.
func (d *KeyspanDecoder) DebugString() string {
	f := binfmt.New(d.blockDecoder.data).LineWidth(20)
	tp := treeprinter.New()
	d.Describe(f, tp.Child("keyspan-decoder"))
	return tp.String()
}

// Describe describes the binary format of the keyspan block, assuming
// f.Offset() is positioned at the beginning of the same keyspan block described
// by r.
func (d *KeyspanDecoder) Describe(f *binfmt.Formatter, tp treeprinter.Node) {
	// Set the relative offset. When loaded into memory, the beginning of blocks
	// are aligned. Padding that ensures alignment is done relative to the
	// current offset. Setting the relative offset ensures that if we're
	// describing this block within a larger structure (eg, f.Offset()>0), we
	// compute padding appropriately assuming the current byte f.Offset() is
	// aligned.
	f.SetAnchorOffset()

	n := tp.Child("keyspan block header")
	f.HexBytesln(4, "user key count: %d", d.boundaryKeysCount)
	f.ToTreePrinter(n)
	d.blockDecoder.HeaderToBinFormatter(f, n)

	for i := 0; i < keyspanColumnCount; i++ {
		// Not all columns in a keyspan block have the same number of rows; the
		// boundary columns columns are different (and their lengths are held in
		// the keyspan block header that precedes the ordinary columnar block
		// header).
		rows := int(d.blockDecoder.header.Rows)
		if i == keyspanColBoundaryUserKeys || i == keyspanColBoundaryKeyIndices {
			rows = int(d.boundaryKeysCount)
		}
		d.blockDecoder.ColumnToBinFormatter(f, n, i, rows)
	}
	f.HexBytesln(1, "block padding byte")
	f.ToTreePrinter(n)
}

// searchBoundaryKeys returns the index of the first boundary key greater than
// or equal to key and whether or not the key was found exactly.
//
// The seek key is in destination/external space. If a synthetic prefix is set,
// the seek key's leading bytes must match the synthetic prefix or the seek
// falls entirely outside the block. If a BlockPrefixSubstitution is set, the
// seek key's leading bytes must match the substitution's Dst; the iterator
// then translates the remaining suffix into storage space (by prepending Src)
// before performing the binary search against the stored boundary keys.
//
// At most one of syntheticPrefix and substitution may be set; this is the
// invariant enforced when constructing FragmentTransforms.
func (d *KeyspanDecoder) searchBoundaryKeysTransformed(
	cmp base.Compare,
	key []byte,
	syntheticPrefix blockiter.SyntheticPrefix,
	substitution blockiter.BlockPrefixSubstitution,
	seekKeyBuf *[]byte,
) (index int, equal bool) {
	if syntheticPrefix.IsSet() {
		// The seek key must have the synthetic prefix, otherwise it falls entirely
		// before or after the block's boundary keys.
		var keyPrefix []byte
		keyPrefix, key = splitKey(key, len(syntheticPrefix))
		if cmp := bytes.Compare(keyPrefix, syntheticPrefix); cmp != 0 {
			if cmp < 0 {
				return 0, false
			}
			return int(d.boundaryKeysCount), false
		}
	} else if substitution.IsSet() {
		var keyPrefix []byte
		keyPrefix, key = splitKey(key, len(substitution.Dst))
		if cmp := bytes.Compare(keyPrefix, substitution.Dst); cmp != 0 {
			if cmp < 0 {
				return 0, false
			}
			return int(d.boundaryKeysCount), false
		}
		// Translate from destination-prefix space to storage-prefix space by
		// prepending Src in place of the (already-stripped) Dst.
		buf := (*seekKeyBuf)[:0]
		buf = append(buf, substitution.Src...)
		buf = append(buf, key...)
		*seekKeyBuf = buf
		key = buf
	}

	i, j := 0, int(d.boundaryKeysCount)
	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		switch cmp(key, d.boundaryKeys.At(h)) {
		case +1:
			i = h + 1
		case 0:
			return h, true
		default:
			// -1
			j = h
		}
	}
	return i, false
}

// NewKeyspanIter constructs a new iterator over a keyspan columnar block.
func NewKeyspanIter(
	cmp base.Compare, h block.BufferHandle, transforms blockiter.FragmentTransforms,
) *KeyspanIter {
	i := keyspanIterPool.Get().(*KeyspanIter)
	i.closeCheck = invariants.CloseChecker{}
	i.handle = h
	d := (*KeyspanDecoder)(unsafe.Pointer(h.BlockMetadata()))
	i.init(cmp, d, transforms)
	return i
}

var keyspanIterPool = sync.Pool{
	New: func() interface{} {
		i := &KeyspanIter{}
		if invariants.UseFinalizers {
			invariants.SetFinalizer(i, func(obj interface{}) {
				if i := obj.(*KeyspanIter); i.handle.Valid() {
					fmt.Fprintf(os.Stderr, "KeyspanIter.handle is not nil: %#v\n", i.handle)
					os.Exit(1)
				}
			})
		}
		return i
	},
}

// A KeyspanIter is an iterator over a columnar keyspan block. It implements the
// keyspan.FragmentIterator interface.
type KeyspanIter struct {
	keyspanIter
	handle block.BufferHandle

	closeCheck invariants.CloseChecker
}

// Close closes the iterator.
func (i *KeyspanIter) Close() {
	i.handle.Release()
	i.handle = block.BufferHandle{}

	if invariants.Sometimes(25) {
		// In invariants mode, sometimes don't add the object to the pool so
		// that we can check for double closes that take longer than the object
		// stays in the pool.
		return
	}

	i.keyspanIter.Close()
	i.closeCheck.Close()
	keyspanIterPool.Put(i)
}

// A keyspanIter is an iterator over a keyspan block. It implements the
// keyspan.FragmentIterator interface.
type keyspanIter struct {
	r            *KeyspanDecoder
	cmp          base.Compare
	transforms   blockiter.FragmentTransforms
	noTransforms bool
	span         keyspan.Span
	// When positioned, the current span's start key is the user key at
	//   i.r.userKeys.At(i.startBoundIndex)
	// and the current span's end key is the user key at
	//   i.r.userKeys.At(i.startBoundIndex+1)
	startBoundIndex int
	keyBuf          [2]keyspan.Key
	// startKeyBuf and endKeyBuf are used when transforms.SyntheticPrefix or
	// transforms.BlockPrefixSubstitution is set.
	startKeyBuf []byte
	endKeyBuf   []byte
	// seekKeyBuf is a scratch buffer used to translate a seek key from
	// destination space to storage space when transforms.BlockPrefixSubstitution
	// is set. It is reused across Seek calls; consumers of the boundary index
	// returned by searchBoundaryKeysTransformed must not retain references into
	// this buffer.
	seekKeyBuf []byte
}

// Assert that KeyspanIter implements the FragmentIterator interface.
var _ keyspan.FragmentIterator = (*keyspanIter)(nil)

// init initializes the iterator with the given comparison function and keyspan
// decoder.
func (i *keyspanIter) init(
	cmp base.Compare, r *KeyspanDecoder, transforms blockiter.FragmentTransforms,
) {
	i.r = r
	i.cmp = cmp
	i.transforms = transforms
	i.noTransforms = transforms.NoTransforms()
	i.span.Start, i.span.End = nil, nil
	i.startBoundIndex = -1
	if i.span.Keys == nil {
		i.span.Keys = i.keyBuf[:0]
	}
	i.startKeyBuf = i.startKeyBuf[:0]
	i.endKeyBuf = i.endKeyBuf[:0]
	i.seekKeyBuf = i.seekKeyBuf[:0]
	if transforms.HasSyntheticPrefix() {
		i.startKeyBuf = append(i.startKeyBuf, transforms.SyntheticPrefix()...)
		i.endKeyBuf = append(i.endKeyBuf, transforms.SyntheticPrefix()...)
	} else if transforms.BlockPrefixSubstitution.IsSet() {
		// Pre-fill the start/end key buffers with the destination prefix; the
		// remainder of each emitted boundary key (after stripping Src) will be
		// appended in materializeSpan.
		i.startKeyBuf = append(i.startKeyBuf, transforms.BlockPrefixSubstitution.Dst...)
		i.endKeyBuf = append(i.endKeyBuf, transforms.BlockPrefixSubstitution.Dst...)
	}
}

func spanStr(s *keyspan.Span) string {
	if s == nil || !s.Valid() {
		return "<nil>"
	}
	return s.String()
}

// SeekGE moves the iterator to the first span covering a key greater than
// or equal to the given key. This is equivalent to seeking to the first
// span with an end key greater than the given key.
func (i *keyspanIter) SeekGE(key []byte) (span *keyspan.Span, _ error) {
	if treesteps.Enabled && treesteps.IsRecording(i) {
		op := treesteps.StartOpf(i, "SeekGE(%q)", key)
		defer func() {
			op.Finishf("= %s", spanStr(span))
		}()
	}

	// Seek among the boundary keys.
	j, eq := i.r.searchBoundaryKeysTransformed(
		i.cmp, key, i.transforms.SyntheticPrefix(),
		i.transforms.BlockPrefixSubstitution, &i.seekKeyBuf)
	// If the found boundary key does not exactly equal the given key, it's
	// strictly greater than key. We need to back up one to consider the span
	// that ends at the this boundary key.
	if !eq {
		j = max(j-1, 0)
	}
	return i.gatherKeysForward(j), nil
}

// SeekLT moves the iterator to the last span covering a key less than the
// given key. This is equivalent to seeking to the last span with a start
// key less than the given key.
func (i *keyspanIter) SeekLT(key []byte) (span *keyspan.Span, _ error) {
	if treesteps.Enabled && treesteps.IsRecording(i) {
		op := treesteps.StartOpf(i, "SeekLT(%q)", key)
		defer func() {
			op.Finishf("= %s", spanStr(span))
		}()
	}
	j, _ := i.r.searchBoundaryKeysTransformed(
		i.cmp, key, i.transforms.SyntheticPrefix(),
		i.transforms.BlockPrefixSubstitution, &i.seekKeyBuf)
	// searchBoundaryKeys seeks to the first boundary key greater than or equal
	// to key. The span beginning at the boundary key j necessarily does NOT
	// cover any key less < key (it only contains keys ≥ key). Back up one to
	// the first span that begins before [key], or to -1 if there is no such
	// span.
	j--

	// If all boundaries are less than [key], or only the last boundary is
	// greater than the key, then we want the last span so we clamp the index to
	// the second to last boundary.
	return i.gatherKeysBackward(min(j, int(i.r.boundaryKeysCount)-2)), nil
}

// First moves the iterator to the first span.
func (i *keyspanIter) First() (span *keyspan.Span, _ error) {
	if treesteps.Enabled && treesteps.IsRecording(i) {
		op := treesteps.StartOpf(i, "First()")
		defer func() {
			op.Finishf("= %s", spanStr(span))
		}()
	}
	return i.gatherKeysForward(0), nil
}

// Last moves the iterator to the last span.
func (i *keyspanIter) Last() (span *keyspan.Span, _ error) {
	if treesteps.Enabled && treesteps.IsRecording(i) {
		op := treesteps.StartOpf(i, "Last()")
		defer func() {
			op.Finishf("= %s", spanStr(span))
		}()
	}
	return i.gatherKeysBackward(int(i.r.boundaryKeysCount) - 2), nil
}

// Next moves the iterator to the next span.
func (i *keyspanIter) Next() (span *keyspan.Span, _ error) {
	if treesteps.Enabled && treesteps.IsRecording(i) {
		op := treesteps.StartOpf(i, "Next()")
		defer func() {
			op.Finishf("= %s", spanStr(span))
		}()
	}
	return i.gatherKeysForward(i.startBoundIndex + 1), nil
}

// Prev moves the iterator to the previous span.
func (i *keyspanIter) Prev() (span *keyspan.Span, _ error) {
	if treesteps.Enabled && treesteps.IsRecording(i) {
		op := treesteps.StartOpf(i, "Prev()")
		defer func() {
			op.Finishf("= %s", spanStr(span))
		}()
	}
	return i.gatherKeysBackward(max(i.startBoundIndex-1, -1)), nil
}

// gatherKeysForward returns the first non-empty Span in the forward direction,
// starting with the span formed by using the boundary key at index
// [startBoundIndex] as the span's start boundary.
func (i *keyspanIter) gatherKeysForward(startBoundIndex int) *keyspan.Span {
	if invariants.Enabled && startBoundIndex < 0 {
		panic(errors.AssertionFailedf("out of bounds: i.startBoundIndex=%d", errors.Safe(startBoundIndex)))
	}
	i.startBoundIndex = startBoundIndex
	if i.startBoundIndex >= int(i.r.boundaryKeysCount)-1 {
		return nil
	}
	if !i.isNonemptySpan(i.startBoundIndex) {
		if i.startBoundIndex == int(i.r.boundaryKeysCount)-2 {
			// Corruption error
			panic(base.CorruptionErrorf("keyspan block has empty span at end"))
		}
		i.startBoundIndex++
		if !i.isNonemptySpan(i.startBoundIndex) {
			panic(base.CorruptionErrorf("keyspan block has consecutive empty spans"))
		}
	}
	// When BlockPrefixSubstitution is configured, spans whose start key does
	// not have Src as prefix are outside the substituted region (they belong
	// to a different prefix in the underlying physical sstable). Advance past
	// them so that materializeSpan never sees them and never tries to apply
	// the substitution to a non-Src key.
	if sub := i.transforms.BlockPrefixSubstitution; sub.IsSet() {
		for i.startBoundIndex < int(i.r.boundaryKeysCount)-1 {
			if bytes.HasPrefix(i.r.boundaryKeys.At(i.startBoundIndex), sub.Src) {
				break
			}
			i.startBoundIndex++
			if i.startBoundIndex < int(i.r.boundaryKeysCount)-1 && !i.isNonemptySpan(i.startBoundIndex) {
				// Skip empty span between distinct prefixes.
				continue
			}
		}
		if i.startBoundIndex >= int(i.r.boundaryKeysCount)-1 {
			return nil
		}
	}
	return i.materializeSpan()
}

// gatherKeysBackward returns the first non-empty Span in the backward direction,
// starting with the span formed by using the boundary key at index
// [startBoundIndex] as the span's start boundary.
func (i *keyspanIter) gatherKeysBackward(startBoundIndex int) *keyspan.Span {
	i.startBoundIndex = startBoundIndex
	if i.startBoundIndex < 0 {
		return nil
	}
	if invariants.Enabled && i.startBoundIndex >= int(i.r.boundaryKeysCount)-1 {
		panic(errors.AssertionFailedf("out of bounds: i.startBoundIndex=%d, i.r.boundaryKeysCount=%d",
			errors.Safe(i.startBoundIndex), errors.Safe(i.r.boundaryKeysCount)))
	}
	if !i.isNonemptySpan(i.startBoundIndex) {
		if i.startBoundIndex == 0 {
			// Corruption error
			panic(base.CorruptionErrorf("keyspan block has empty span at beginning"))
		}
		i.startBoundIndex--
		if !i.isNonemptySpan(i.startBoundIndex) {
			panic(base.CorruptionErrorf("keyspan block has consecutive empty spans"))
		}
	}
	// See gatherKeysForward: when BlockPrefixSubstitution is configured,
	// retreat past spans whose start key is not in Src-prefix space.
	if sub := i.transforms.BlockPrefixSubstitution; sub.IsSet() {
		for i.startBoundIndex >= 0 {
			if bytes.HasPrefix(i.r.boundaryKeys.At(i.startBoundIndex), sub.Src) {
				break
			}
			i.startBoundIndex--
			if i.startBoundIndex >= 0 && !i.isNonemptySpan(i.startBoundIndex) {
				continue
			}
		}
		if i.startBoundIndex < 0 {
			return nil
		}
	}
	return i.materializeSpan()
}

// isNonemptySpan returns true if the span starting at i.startBoundIndex
// contains keys.
func (i *keyspanIter) isNonemptySpan(startBoundIndex int) bool {
	return i.r.boundaryKeyIndices.At(startBoundIndex) < i.r.boundaryKeyIndices.At(startBoundIndex+1)
}

// materializeSpan constructs the current span from i.startBoundIndex and
// i.{start,end}KeyIndex.
func (i *keyspanIter) materializeSpan() *keyspan.Span {
	i.span = keyspan.Span{
		Start: i.r.boundaryKeys.At(i.startBoundIndex),
		End:   i.r.boundaryKeys.At(i.startBoundIndex + 1),
		Keys:  i.span.Keys[:0],
	}
	startIndex := i.r.boundaryKeyIndices.At(i.startBoundIndex)
	endIndex := i.r.boundaryKeyIndices.At(i.startBoundIndex + 1)
	if cap(i.span.Keys) < int(endIndex-startIndex) {
		i.span.Keys = make([]keyspan.Key, 0, int(endIndex-startIndex))
	}
	for j := startIndex; j < endIndex; j++ {
		i.span.Keys = append(i.span.Keys, keyspan.Key{
			Trailer: base.InternalKeyTrailer(i.r.trailers.At(int(j))),
			Suffix:  i.r.suffixes.At(int(j)),
			Value:   i.r.values.At(int(j)),
		})
	}
	if i.noTransforms {
		return &i.span
	}
	if i.transforms.SyntheticSeqNum != blockiter.NoSyntheticSeqNum {
		for j := range i.span.Keys {
			i.span.Keys[j].Trailer = base.MakeTrailer(
				base.SeqNum(i.transforms.SyntheticSeqNum), i.span.Keys[j].Trailer.Kind())
		}
	}
	if i.transforms.HasSyntheticSuffix() {
		for j := range i.span.Keys {
			k := &i.span.Keys[j]
			switch k.Kind() {
			case base.InternalKeyKindRangeKeySet:
				if len(k.Suffix) > 0 {
					// TODO(jackson): Assert synthetic suffix is >= k.Suffix.
					k.Suffix = i.transforms.SyntheticSuffix()
				}
			case base.InternalKeyKindRangeKeyDelete:
				// Nothing to do.
			default:
				panic(base.AssertionFailedf("synthetic suffix not supported with key kind %s", k.Kind()))
			}
		}
	}
	if sub := i.transforms.BlockPrefixSubstitution; sub.IsSet() {
		// BlockPrefixSubstitution and SyntheticPrefix are mutually exclusive
		// (enforced by FragmentIterTransforms construction); take this branch
		// in preference to the SyntheticPrefix branch below so the
		// invariants.Sometimes path doesn't accidentally bypass substitution.
		// The stored boundary keys include Src as their leading bytes; strip
		// Src and prepend Dst into the pre-allocated start/end key buffers.
		// init() pre-filled the first len(Dst) bytes with Dst, so we only need
		// to truncate back to len(Dst) and append the post-Src tail.
		dstLen := len(sub.Dst)
		i.startKeyBuf = i.startKeyBuf[:dstLen]
		i.endKeyBuf = i.endKeyBuf[:dstLen]
		if invariants.Enabled {
			if !bytes.Equal(i.startKeyBuf, sub.Dst) {
				panic(errors.AssertionFailedf("keyspanIter: substitution Dst mismatch %q, %q",
					i.startKeyBuf, sub.Dst))
			}
			if !bytes.Equal(i.endKeyBuf, sub.Dst) {
				panic(errors.AssertionFailedf("keyspanIter: substitution Dst mismatch %q, %q",
					i.endKeyBuf, sub.Dst))
			}
			if !bytes.HasPrefix(i.span.Start, sub.Src) {
				panic(errors.AssertionFailedf(
					"keyspanIter: span.Start %q does not have substitution Src %q",
					i.span.Start, sub.Src))
			}
			if !bytes.HasPrefix(i.span.End, sub.Src) {
				panic(errors.AssertionFailedf(
					"keyspanIter: span.End %q does not have substitution Src %q",
					i.span.End, sub.Src))
			}
		}
		i.startKeyBuf = append(i.startKeyBuf, i.span.Start[len(sub.Src):]...)
		i.endKeyBuf = append(i.endKeyBuf, i.span.End[len(sub.Src):]...)
		i.span.Start = i.startKeyBuf
		i.span.End = i.endKeyBuf
	} else if i.transforms.HasSyntheticPrefix() || invariants.Sometimes(10) {
		syntheticPrefix := i.transforms.SyntheticPrefix()
		i.startKeyBuf = i.startKeyBuf[:len(syntheticPrefix)]
		i.endKeyBuf = i.endKeyBuf[:len(syntheticPrefix)]
		if invariants.Enabled {
			if !bytes.Equal(i.startKeyBuf, syntheticPrefix) {
				panic(errors.AssertionFailedf("keyspanIter: synthetic prefix mismatch %q, %q",
					i.startKeyBuf, syntheticPrefix))
			}
			if !bytes.Equal(i.endKeyBuf, syntheticPrefix) {
				panic(errors.AssertionFailedf("keyspanIter: synthetic prefix mismatch %q, %q",
					i.endKeyBuf, syntheticPrefix))
			}
		}
		i.startKeyBuf = append(i.startKeyBuf, i.span.Start...)
		i.endKeyBuf = append(i.endKeyBuf, i.span.End...)
		i.span.Start = i.startKeyBuf
		i.span.End = i.endKeyBuf
	}

	return &i.span
}

// Close closes the iterator.
func (i *keyspanIter) Close() {
	*i = keyspanIter{}
}

// SetContext implements keyspan.FragmentIterator.
func (i *keyspanIter) SetContext(context.Context) {}

// WrapChildren implements keyspan.FragmentIterator.
func (i *keyspanIter) WrapChildren(keyspan.WrapFn) {}

// TreeStepsNode is part of the FragmentIterator interface.
func (i *keyspanIter) TreeStepsNode() treesteps.NodeInfo {
	return treesteps.NodeInfof(i, "colblk.keyspanIter")
}
