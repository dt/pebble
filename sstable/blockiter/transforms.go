// Copyright 2025 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package blockiter

import (
	"bytes"
	"unsafe"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/base"
)

// Transforms allow on-the-fly transformation of data at iteration time.
//
// These transformations could in principle be implemented as block transforms
// (at least for non-virtual sstables), but applying them during iteration is
// preferable.
type Transforms struct {
	// SyntheticSeqNum, if set, overrides the sequence number in all keys. It is
	// set if the sstable was ingested or it is foreign.
	SyntheticSeqNum SyntheticSeqNum
	// HideObsoletePoints, if true, skips over obsolete points during iteration.
	// This is the norm when the sstable is foreign or the largest sequence number
	// of the sstable is below the one we are reading.
	HideObsoletePoints bool

	SyntheticPrefixAndSuffix SyntheticPrefixAndSuffix

	// BlockPrefixSubstitution, if set, replaces a leading source prefix with a
	// destination prefix during key materialization. Mutually exclusive with
	// SyntheticPrefixAndSuffix's prefix component. See BlockPrefixSubstitution.
	BlockPrefixSubstitution BlockPrefixSubstitution
}

// NoTransforms is the default value for Transforms.
var NoTransforms = Transforms{}

// NoTransforms returns true if there are no transforms enabled.
func (t *Transforms) NoTransforms() bool {
	return t.SyntheticSeqNum == 0 &&
		!t.HideObsoletePoints &&
		t.SyntheticPrefixAndSuffix.IsUnset() &&
		!t.BlockPrefixSubstitution.IsSet()
}

func (t *Transforms) HasSyntheticPrefix() bool {
	return t.SyntheticPrefixAndSuffix.HasPrefix()
}

func (t *Transforms) SyntheticPrefix() []byte {
	return t.SyntheticPrefixAndSuffix.Prefix()
}

func (t *Transforms) HasSyntheticSuffix() bool {
	return t.SyntheticPrefixAndSuffix.HasSuffix()
}

func (t *Transforms) SyntheticSuffix() []byte {
	return t.SyntheticPrefixAndSuffix.Suffix()
}

// FragmentTransforms allow on-the-fly transformation of range deletion or
// range key data at iteration time.
//
// BlockPrefixSubstitution is applied by the colblk fragment iterator when
// emitting fragment user keys (Apply on emit) and inverted on seek keys
// (Invert on seek), mirroring how the colblk data-block iterator handles the
// substitution. The row-based fragment iterator does not implement the
// substitution; the only consumer that sets BlockPrefixSubstitution
// (VirtualClone) rejects row-based source SSTs before constructing the
// transform.
type FragmentTransforms struct {
	SyntheticSeqNum          SyntheticSeqNum
	SyntheticPrefixAndSuffix SyntheticPrefixAndSuffix

	// BlockPrefixSubstitution, if set, replaces a leading source prefix with a
	// destination prefix during fragment user-key materialization (start, end).
	// Mutually exclusive with SyntheticPrefixAndSuffix's prefix component. See
	// BlockPrefixSubstitution.
	BlockPrefixSubstitution BlockPrefixSubstitution
}

// NoTransforms returns true if there are no transforms enabled.
func (t *FragmentTransforms) NoTransforms() bool {
	return t.SyntheticSeqNum == 0 &&
		t.SyntheticPrefixAndSuffix.IsUnset() &&
		!t.BlockPrefixSubstitution.IsSet()
}

func (t *FragmentTransforms) HasSyntheticPrefix() bool {
	return t.SyntheticPrefixAndSuffix.HasPrefix()
}

func (t *FragmentTransforms) SyntheticPrefix() []byte {
	return t.SyntheticPrefixAndSuffix.Prefix()
}

func (t *FragmentTransforms) HasSyntheticSuffix() bool {
	return t.SyntheticPrefixAndSuffix.HasSuffix()
}

func (t *FragmentTransforms) SyntheticSuffix() []byte {
	return t.SyntheticPrefixAndSuffix.Suffix()
}

// NoFragmentTransforms is the default value for Transforms.
var NoFragmentTransforms = FragmentTransforms{}

// SyntheticSeqNum is used to override all sequence numbers in a table. It is
// set to a non-zero value when the table was created externally and ingested
// whole.
type SyntheticSeqNum base.SeqNum

// NoSyntheticSeqNum is the default zero value for SyntheticSeqNum, which
// disables overriding the sequence number.
const NoSyntheticSeqNum SyntheticSeqNum = 0

// SyntheticSuffix will replace every suffix of every point key surfaced during
// block iteration. A synthetic suffix can be used if:
//  1. no two keys in the sst share the same prefix; and
//  2. pebble.Compare(prefix + replacementSuffix, prefix + originalSuffix) < 0,
//     for all keys in the backing sst which have a suffix (i.e. originalSuffix
//     is not empty).
//
// Range dels are not supported when synthetic suffix is used.
//
// For range keys, the synthetic suffix applies to the suffix that is part of
// RangeKeySet - if it is non-empty, it is replaced with the SyntheticSuffix.
// RangeKeyUnset keys are not supported when a synthetic suffix is used.
type SyntheticSuffix []byte

// IsSet returns true if the synthetic suffix is not empty.
func (ss SyntheticSuffix) IsSet() bool {
	return len(ss) > 0
}

// SyntheticPrefix represents a byte slice that is implicitly prepended to every
// key in a file being read or accessed by a reader. Note that since the byte
// slice is prepended to every KV rather than replacing a byte prefix, the
// result of prepending the synthetic prefix must be a full, valid key while the
// partial key physically stored within the sstable need not be a valid key
// according to user key semantics.
//
// Note that elsewhere we use the language of 'prefix' to describe the user key
// portion of a MVCC key, as defined by the Comparer's base.Split method. The
// SyntheticPrefix is related only in that it's a byte prefix that is
// incorporated into the logical MVCC prefix.
//
// The table's bloom filters are constructed only on the partial keys physically
// stored in the table, but interactions with the file including seeks and
// reads will all behave as if the file had been constructed from keys that
// include the synthetic prefix. Note that all Compare operations will act on a
// partial key (before any prepending), so the Comparer must support comparing
// these partial keys.
//
// The synthetic prefix will never modify key metadata stored in the key suffix.
//
// NB: Since this transformation currently only applies to point keys, a block
// with range keys cannot be iterated over with a synthetic prefix.
type SyntheticPrefix []byte

// IsSet returns true if the synthetic prefix is not enpty.
func (sp SyntheticPrefix) IsSet() bool {
	return len(sp) > 0
}

// Apply prepends the synthetic prefix to a key.
func (sp SyntheticPrefix) Apply(key []byte) []byte {
	res := make([]byte, 0, len(sp)+len(key))
	res = append(res, sp...)
	res = append(res, key...)
	return res
}

// Invert removes the synthetic prefix from a key.
func (sp SyntheticPrefix) Invert(key []byte) []byte {
	res, ok := bytes.CutPrefix(key, sp)
	if !ok {
		panic(errors.AssertionFailedf("unexpected prefix: %s", key))
	}
	return res
}

// SyntheticPrefixAndSuffix is a more compact way of representing both a
// synthetic prefix and a synthetic suffix. See SyntheticPrefix and
// SyntheticSuffix.
//
// The zero value is valid, representing no synthetic prefix or suffix.
type SyntheticPrefixAndSuffix struct {
	prefixLen uint32
	suffixLen uint32
	// buf is either nil (iff prefixLen=suffixLen=0) or a pointer to a buffer
	// containing the prefix followed by the suffix.
	buf unsafe.Pointer
}

// MakeSyntheticPrefixAndSuffix returns a SyntheticPrefixAndSuffix with the
// given prefix and suffix.
func MakeSyntheticPrefixAndSuffix(
	prefix SyntheticPrefix, suffix SyntheticSuffix,
) SyntheticPrefixAndSuffix {
	if !prefix.IsSet() && !suffix.IsSet() {
		return SyntheticPrefixAndSuffix{}
	}
	buf := make([]byte, len(prefix)+len(suffix))
	copy(buf, prefix)
	copy(buf[len(prefix):], suffix)
	return SyntheticPrefixAndSuffix{
		prefixLen: uint32(len(prefix)),
		suffixLen: uint32(len(suffix)),
		buf:       unsafe.Pointer(&buf[0]),
	}
}

// IsUnset returns true if HasPrefix() and HasSuffix() both return false.
func (ps SyntheticPrefixAndSuffix) IsUnset() bool {
	return ps.buf == nil
}

// HasPrefix returns true if ps contains a non-empty synthetic prefix.
func (ps SyntheticPrefixAndSuffix) HasPrefix() bool {
	return ps.prefixLen != 0
}

// PrefixLen returns the length of the synthetic prefix, or 0 if it is not set.
func (ps SyntheticPrefixAndSuffix) PrefixLen() uint32 {
	return ps.prefixLen
}

// Prefix returns the synthetic prefix.
func (ps SyntheticPrefixAndSuffix) Prefix() SyntheticPrefix {
	if ps.prefixLen == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(ps.buf), ps.prefixLen)
}

// HasSuffix returns true if ps contains a non-empty synthetic suffix.
func (ps SyntheticPrefixAndSuffix) HasSuffix() bool {
	return ps.suffixLen != 0
}

// SuffixLen returns the length of the synthetic prefix, or 0 if it is not set.
func (ps SyntheticPrefixAndSuffix) SuffixLen() uint32 {
	return ps.suffixLen
}

// Suffix returns the synthetic suffix.
func (ps SyntheticPrefixAndSuffix) Suffix() SyntheticSuffix {
	if ps.suffixLen == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(uintptr(ps.buf)+uintptr(ps.prefixLen))), ps.suffixLen)
}

// RemoveSuffix returns a SyntheticPrefixAndSuffix that has the same prefix as
// the receiver but no suffix.
func (ps SyntheticPrefixAndSuffix) RemoveSuffix() SyntheticPrefixAndSuffix {
	if ps.prefixLen == 0 {
		return SyntheticPrefixAndSuffix{}
	}
	return SyntheticPrefixAndSuffix{
		prefixLen: ps.prefixLen,
		suffixLen: 0,
		buf:       ps.buf,
	}
}

// BlockPrefixSubstitution describes a substitution applied to the block-shared
// prefix of every colblk data block when reading a virtual sstable. The first
// len(Src) bytes at the start of the stored block-shared prefix are replaced
// with Dst during key materialization.
//
// Precondition: every block touched by iteration must have a stored shared
// prefix whose first len(Src) bytes equal Src. For fully-contained virtual
// SSTs this is guaranteed by the bounds invariant: the LCP of the SST's
// smallest and largest keys is a superset of any prefix the SST's bounds lie
// within, and every block's stored shared prefix is itself a superset of
// LCP(smallest, largest). For block-aligned virtual SSTs over straddling
// files, the bounds exclude blocks where the precondition would fail.
//
// Unlike SyntheticPrefix (which is *prepended* to keys whose backing SST
// stores them with the prefix already stripped), BlockPrefixSubstitution
// operates on SSTs whose physically stored keys retain Src as the leading
// bytes of every key. The substitution is applied once per block at the
// block-shared-prefix level, not per key, so per-key iteration cost is zero.
//
// At seek time, callers in the destination key space must Invert the seek
// key (replace Dst with Src) before consulting the block's index/search
// structures; the iterator-emitted keys are produced via Apply.
type BlockPrefixSubstitution struct {
	// Src is the byte slice present at the start of every block-shared prefix
	// in the underlying sstable that this transform is configured to replace.
	Src []byte
	// Dst is the byte slice substituted in place of Src.
	Dst []byte
}

// IsSet returns true if the substitution is configured to strip and replace a
// non-empty source prefix. A non-empty Src is the load-bearing precondition
// for the transform: without Src there is nothing to substitute, so a
// substitution with an empty Src and a non-empty Dst would degenerate into a
// pure prepend. If you want to prepend without stripping, use SyntheticPrefix
// instead — that's the supported mechanism for that case. Disallowing the
// "empty Src + non-empty Dst" encoding keeps the mutual-exclusion check
// against SyntheticPrefix meaningful.
func (s BlockPrefixSubstitution) IsSet() bool {
	return len(s.Src) > 0
}

// Apply transforms a storage-space key into a destination-space key by
// stripping the leading len(Src) bytes (which must equal Src) and prepending
// Dst.
func (s BlockPrefixSubstitution) Apply(storedKey []byte) []byte {
	if len(s.Src) == 0 {
		panic(errors.AssertionFailedf("BlockPrefixSubstitution.Apply called with empty Src; use SyntheticPrefix for pure prepend"))
	}
	if !bytes.HasPrefix(storedKey, s.Src) {
		panic(errors.AssertionFailedf("stored key %q does not have expected source prefix %q", storedKey, s.Src))
	}
	res := make([]byte, 0, len(s.Dst)+len(storedKey)-len(s.Src))
	res = append(res, s.Dst...)
	res = append(res, storedKey[len(s.Src):]...)
	return res
}

// Invert transforms a destination-space key into a storage-space key by
// stripping the leading len(Dst) bytes (which must equal Dst) and prepending
// Src.
func (s BlockPrefixSubstitution) Invert(externalKey []byte) []byte {
	rest, ok := bytes.CutPrefix(externalKey, s.Dst)
	if !ok {
		panic(errors.AssertionFailedf("external key %q does not have expected destination prefix %q", externalKey, s.Dst))
	}
	res := make([]byte, 0, len(s.Src)+len(rest))
	res = append(res, s.Src...)
	res = append(res, rest...)
	return res
}
