package virtual

import "github.com/cockroachdb/pebble/internal/base"

// VirtualReaderParams are the parameters necessary for a reader to read virtual sstables.
type VirtualReaderParams struct {
	Lower   base.InternalKey
	Upper   base.InternalKey
	FileNum base.FileNum
}

// Constrain bounds will narrow the start, end bounds if they do not fit within
// the virtual sstable. The function will return if the new end key is
// inclusive.
//
// Note on BlockPrefixSubstitution: when the virtual sstable has a
// BlockPrefixSubstitution configured, the externally-visible bounds (v.Lower,
// v.Upper) and the start/end inputs are all in destination-prefix space. The
// returned bounds remain in destination-prefix space; no inversion to source
// space is performed here. The data-block iterator is responsible for
// inverting destination-space seek keys to source space before consulting the
// underlying block (see colblk.DataBlockIter.seekGEInternal). Iterator-emitted
// keys are produced via BlockPrefixSubstitution.Apply, so per-key bounds
// comparisons against the returned upper/lower in singleLevelIterator are
// also in destination space and consistent.
func (v *VirtualReaderParams) ConstrainBounds(
	start, end []byte, endInclusive bool, compare func([]byte, []byte) int,
) (lastKeyInclusive bool, first []byte, last []byte) {
	first = start
	if start == nil || compare(start, v.Lower.UserKey) < 0 {
		first = v.Lower.UserKey
	}

	// Note that we assume that start, end has some overlap with the virtual
	// sstable bounds.
	last = v.Upper.UserKey
	lastKeyInclusive = !v.Upper.IsExclusiveSentinel()
	if end != nil {
		cmp := compare(end, v.Upper.UserKey)
		switch {
		case cmp == 0:
			lastKeyInclusive = !v.Upper.IsExclusiveSentinel() && endInclusive
			last = v.Upper.UserKey
		case cmp > 0:
			lastKeyInclusive = !v.Upper.IsExclusiveSentinel()
			last = v.Upper.UserKey
		default:
			lastKeyInclusive = endInclusive
			last = end
		}
	}
	// TODO(bananabrick): What if someone passes in bounds completely outside of
	// virtual sstable bounds?
	return lastKeyInclusive, first, last
}
