// Copyright 2026 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package pebble

import (
	"bytes"
	"context"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/internal/manifest"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/sstable/block"
)

// ErrUnsupportedClone is returned by VirtualClone when it encounters an LSM
// shape that this version of Pebble cannot clone via the substitution
// mechanism. The error is wrapped with details describing why the clone could
// not be performed.
//
// Cases that currently return ErrUnsupportedClone:
//   - A source sstable that uses the row-based block format intersects
//     srcSpan.
//   - A source sstable that contains range keys intersects srcSpan.
//   - A source sstable straddles srcSpan (it extends beyond the in-span
//     region). A future slice will rewrite the boundary blocks to support
//     this case.
//   - The destination region in the LSM is not empty at the level the source
//     occupies. A future slice will implement smarter level placement.
var ErrUnsupportedClone = errors.New("pebble: unsupported VirtualClone case")

// virtualCloneMaxRetries bounds the number of times VirtualClone will retry
// after losing a race against a concurrent compaction or excise that mutated
// the source backings between the read phase and the apply phase.
const virtualCloneMaxRetries = 5

// VirtualClone walks every SST in the current LSM that intersects srcSpan and
// creates virtual SSTs (sharing existing physical backings) that expose the
// intersected keys under dstPrefix instead of srcPrefix.
//
// srcSpan must lie entirely within [srcPrefix, srcPrefix.ImmediateSuccessor).
// The destination span [dstPrefix + suffix(srcSpan)] is expected to be empty
// in the LSM; if it is not, this slice returns ErrUnsupportedClone. (D1: this
// slice errors on any conflict; sophisticated placement is a TODO.)
//
// Atomicity: a single VersionEdit installs all virtual SSTs. On conflict with
// a concurrent compaction/excise on a referenced source SST, the operation
// restarts (bounded retry).
//
// Returns ErrUnsupportedClone (with details) for v1 unsupported cases:
//   - rowblk-format SSTs intersecting srcSpan
//   - SSTs that straddle srcSpan (i.e. extend beyond the in-span region)
//     [D1: this slice errors on straddlers; D2 will rewrite their boundary
//     blocks]
//   - any source SST containing range keys (range-key path is deferred)
//   - destination region not empty (D1: error; future: smarter placement)
func (d *DB) VirtualClone(
	ctx context.Context, srcSpan KeyRange, srcPrefix, dstPrefix []byte,
) error {
	if err := d.closed.Load(); err != nil {
		panic(err)
	}
	if d.opts.ReadOnly {
		return ErrReadOnly
	}
	if v := d.FormatMajorVersion(); v < FormatPrefixSubstitution {
		return errors.Newf(
			"pebble: VirtualClone requires format major version %d; database is at %d",
			FormatPrefixSubstitution, v,
		)
	}
	if err := validateVirtualCloneInputs(d.opts.Comparer, srcSpan, srcPrefix, dstPrefix); err != nil {
		return err
	}

	for attempt := 0; attempt < virtualCloneMaxRetries; attempt++ {
		retried, err := d.virtualCloneAttempt(ctx, srcSpan, srcPrefix, dstPrefix)
		if err != nil {
			return err
		}
		if !retried {
			return nil
		}
	}
	return errors.Newf("pebble: VirtualClone exhausted %d retries due to concurrent LSM mutations",
		virtualCloneMaxRetries)
}

// validateVirtualCloneInputs checks the static preconditions of VirtualClone.
func validateVirtualCloneInputs(
	cmp *base.Comparer, srcSpan KeyRange, srcPrefix, dstPrefix []byte,
) error {
	if len(srcPrefix) == 0 {
		return errors.New("pebble: VirtualClone requires a non-empty srcPrefix")
	}
	if len(dstPrefix) == 0 {
		return errors.New("pebble: VirtualClone requires a non-empty dstPrefix")
	}
	if bytes.Equal(srcPrefix, dstPrefix) {
		return errors.New("pebble: VirtualClone srcPrefix and dstPrefix must differ")
	}
	if !srcSpan.Valid() {
		return errors.New("pebble: VirtualClone requires a valid srcSpan")
	}
	if cmp.Compare(srcSpan.Start, srcSpan.End) >= 0 {
		return errors.Newf("pebble: VirtualClone srcSpan start %q is not before end %q",
			srcSpan.Start, srcSpan.End)
	}
	// srcSpan must lie entirely within [srcPrefix, srcPrefix.ImmediateSuccessor).
	// We require srcPrefix itself to be a prefix key (Split(srcPrefix) ==
	// len(srcPrefix)) and srcSpan.Start/End must be at or after srcPrefix and
	// at or before srcPrefix.ImmediateSuccessor.
	if cmp.Split == nil || cmp.Split(srcPrefix) != len(srcPrefix) {
		return errors.New("pebble: VirtualClone srcPrefix must be a prefix key (Split(srcPrefix)==len(srcPrefix))")
	}
	if cmp.Split(dstPrefix) != len(dstPrefix) {
		return errors.New("pebble: VirtualClone dstPrefix must be a prefix key (Split(dstPrefix)==len(dstPrefix))")
	}
	// srcSpan.Start must have srcPrefix as a prefix.
	if !bytes.HasPrefix(srcSpan.Start, srcPrefix) {
		return errors.Newf(
			"pebble: VirtualClone srcSpan start %q does not have srcPrefix %q",
			srcSpan.Start, srcPrefix)
	}
	// srcSpan.End must be at most the smallest user key not starting with
	// srcPrefix, so every key inside srcSpan has srcPrefix as its leading
	// bytes. We allow either srcSpan.End to itself start with srcPrefix (a
	// sub-range clone) or srcSpan.End to equal an end-of-prefix marker
	// computed by appending 0xFF bytes / using the comparer's
	// ImmediateSuccessor — since we cannot enforce a single canonical end
	// uniformly across comparers, the per-source-SST check below ensures that
	// each cloned table's bounds actually start with srcPrefix.
	return nil
}

// translateUserKey rewrites a user key in srcPrefix-space into dstPrefix-space.
// The caller guarantees the key has srcPrefix as its leading bytes.
func translateUserKey(srcPrefix, dstPrefix, key []byte) []byte {
	if !bytes.HasPrefix(key, srcPrefix) {
		panic(errors.AssertionFailedf(
			"pebble: VirtualClone: key %q does not start with srcPrefix %q", key, srcPrefix))
	}
	out := make([]byte, 0, len(dstPrefix)+len(key)-len(srcPrefix))
	out = append(out, dstPrefix...)
	out = append(out, key[len(srcPrefix):]...)
	return out
}

// translateInternalKey rewrites an InternalKey from src-space to dst-space,
// preserving the trailer (sequence number and kind).
func translateInternalKey(srcPrefix, dstPrefix []byte, k base.InternalKey) base.InternalKey {
	return base.InternalKey{
		UserKey: translateUserKey(srcPrefix, dstPrefix, k.UserKey),
		Trailer: k.Trailer,
	}
}

// virtualCloneSource describes a single source sstable that intersects
// srcSpan and is fully contained within it. Collected during the read phase
// and consumed by the apply phase.
type virtualCloneSource struct {
	level  int
	source *manifest.TableMetadata
}

// virtualCloneAttempt performs one attempt at VirtualClone. It returns
// (retried=true, nil) if the attempt aborted because the LSM mutated between
// the read phase and the apply phase; the caller may retry. It returns
// (false, err) on terminal error or (false, nil) on success.
func (d *DB) virtualCloneAttempt(
	ctx context.Context, srcSpan KeyRange, srcPrefix, dstPrefix []byte,
) (retried bool, _ error) {
	// Snapshot the current Version + take a ref so the source backings remain
	// alive through our read phase.
	d.mu.Lock()
	currentVersion := d.mu.versions.currentVersion()
	currentVersion.Ref()
	d.mu.Unlock()
	defer currentVersion.Unref()

	srcSpanBounds := srcSpan.UserKeyBounds()

	// Walk the LSM, classifying intersecting source SSTs.
	var sources []virtualCloneSource
	for layer, ls := range currentVersion.AllLevelsAndSublevels() {
		level := layer.Level()
		for m := range ls.Overlaps(d.cmp, srcSpanBounds).All() {
			// Range keys: deferred.
			if m.HasRangeKeys {
				return false, errors.Wrapf(ErrUnsupportedClone,
					"source table %s at L%d intersecting srcSpan contains range keys",
					m.TableNum, level)
			}
			// Reject straddlers: source SST must be fully contained.
			if !srcSpanBounds.ContainsInternalKey(d.cmp, m.Smallest()) ||
				!srcSpanBounds.ContainsInternalKey(d.cmp, m.Largest()) {
				return false, errors.Wrapf(ErrUnsupportedClone,
					"source table %s at L%d straddles srcSpan boundary (smallest=%s largest=%s)",
					m.TableNum, level,
					m.Smallest().Pretty(d.opts.Comparer.FormatKey),
					m.Largest().Pretty(d.opts.Comparer.FormatKey))
			}
			// Defense in depth: confirm that both bound user keys actually start
			// with srcPrefix. The substitution would produce nonsense (or panic)
			// for keys that don't carry srcPrefix as their leading bytes.
			if !bytes.HasPrefix(m.Smallest().UserKey, srcPrefix) ||
				!bytes.HasPrefix(m.Largest().UserKey, srcPrefix) {
				return false, errors.Wrapf(ErrUnsupportedClone,
					"source table %s at L%d has bounds outside srcPrefix %q (smallest=%s largest=%s)",
					m.TableNum, level, srcPrefix,
					m.Smallest().Pretty(d.opts.Comparer.FormatKey),
					m.Largest().Pretty(d.opts.Comparer.FormatKey))
			}
			// Reject row-based table formats. We need to open the file to check
			// its format. Use the file cache without the DB mutex held.
			var isRowblk bool
			if err := d.fileCache.withReader(ctx, block.NoReadEnv, m,
				func(r *sstable.Reader, _ sstable.ReadEnv) error {
					format, err := r.TableFormat()
					if err != nil {
						return err
					}
					isRowblk = !format.BlockColumnar()
					return nil
				}); err != nil {
				return false, errors.Wrapf(err,
					"pebble: VirtualClone failed to read format of source table %s", m.TableNum)
			}
			if isRowblk {
				return false, errors.Wrapf(ErrUnsupportedClone,
					"source table %s at L%d uses the row-based block format",
					m.TableNum, level)
			}
			sources = append(sources, virtualCloneSource{level: level, source: m})
		}
	}

	// Empty src span: no-op.
	if len(sources) == 0 {
		return false, nil
	}

	// Build virtual TableMetadata entries. We allocate TableNums up front; if
	// the apply phase fails these become unused holes in the namespace, which
	// is acceptable (the same is true for ingest's pre-VE physical SSTs).
	type virtualEntry struct {
		level   int
		source  *manifest.TableMetadata
		virtual *manifest.TableMetadata
	}
	entries := make([]virtualEntry, 0, len(sources))
	for _, src := range sources {
		m := src.source
		vm := &manifest.TableMetadata{
			Virtual:               true,
			TableNum:              d.mu.versions.getNextTableNum(),
			SeqNums:               m.SeqNums,
			LargestSeqNumAbsolute: m.LargestSeqNumAbsolute,
			BlockPrefixSubstitution: sstable.BlockPrefixSubstitution{
				Src: append([]byte(nil), srcPrefix...),
				Dst: append([]byte(nil), dstPrefix...),
			},
		}
		// Translate the per-key-type bounds (not the overall bounds) so that
		// HasPointKeys/HasRangeKeys and PointKeyBounds/RangeKeyBounds are
		// populated correctly via ExtendPointKeyBounds/ExtendRangeKeyBounds.
		// We rejected sources with range keys above, so HasPointKeys must be
		// true here.
		if !m.HasPointKeys {
			return false, errors.AssertionFailedf(
				"pebble: VirtualClone source table %s has no point keys", m.TableNum)
		}
		smallest := translateInternalKey(srcPrefix, dstPrefix, m.PointKeyBounds.Smallest())
		largest := translateInternalKey(srcPrefix, dstPrefix, m.PointKeyBounds.Largest())
		vm.ExtendPointKeyBounds(d.cmp, smallest, largest)

		vm.AttachVirtualBacking(m.TableBacking)
		// Size: D1 reuses the source's size as an approximation. The cloned
		// virtual table covers the entirety of the source (fully-contained
		// case), so this is a tight upper bound.
		vm.Size = m.Size
		if vm.Size == 0 {
			vm.Size = 1
		}
		// Blob references are scaled the same way excise does it. For a
		// fully-contained clone the ratio is 1.
		determineExcisedTableBlobReferences(m.BlobReferences, m.Size, vm, d.FormatMajorVersion())

		if err := vm.Validate(d.cmp, d.opts.Comparer.FormatKey); err != nil {
			return false, errors.Wrapf(err,
				"pebble: VirtualClone produced invalid virtual table for source %s", m.TableNum)
		}
		vm.ValidateVirtual(m)

		entries = append(entries, virtualEntry{
			level:   src.level,
			source:  m,
			virtual: vm,
		})
	}

	// Apply via UpdateVersionLocked. The callback re-snapshots the current
	// version and validates that every source backing still exists and is
	// still resident at the level we intend to slot the virtual file at. If
	// not, we abort and request a retry.
	d.mu.Lock()
	jobID := d.newJobIDLocked()
	defer d.mu.Unlock()

	var aborted bool
	_, err := d.mu.versions.UpdateVersionLocked(func() (versionUpdate, error) {
		current := d.mu.versions.currentVersion()

		// Re-validate every source backing still exists in the latest virtual
		// backings set OR the source is still a non-virtual file in current.
		// We also confirm the source is still present at the same level it was
		// observed at; if it has moved, restart so the level placement is
		// recomputed against the new shape.
		for _, e := range entries {
			if !current.Contains(e.level, e.source) {
				aborted = true
				return versionUpdate{}, nil
			}
		}

		// D1 simple level placement: try to place each virtual file at the
		// same level as its source. If anything at that level overlaps the
		// new file's dst-space bounds, return ErrUnsupportedClone.
		for _, e := range entries {
			vmBounds := e.virtual.UserKeyBounds()
			if current.HasOverlap(e.level, vmBounds) {
				return versionUpdate{}, errors.Wrapf(ErrUnsupportedClone,
					"destination region overlaps existing data at L%d for cloned table %s",
					e.level, e.virtual.TableNum)
			}
		}

		// Build the version edit.
		ve := &manifest.VersionEdit{}
		// Track which backings we've already added to CreatedBackingTables in
		// this VE: a single backing might be referenced by multiple cloned
		// virtual files, but per CreatedBackingTables invariants each backing
		// is added at most once per VE. A backing is "new" to the
		// virtualBackings set if it's not currently present (i.e. the source
		// is a non-virtual table).
		seenNewBacking := make(map[base.DiskFileNum]struct{})
		for _, e := range entries {
			ve.NewTables = append(ve.NewTables, manifest.NewTableEntry{
				Level: e.level,
				Meta:  e.virtual,
			})
			backingNum := e.source.TableBacking.DiskFileNum
			if _, ok := d.mu.versions.latest.virtualBackings.Get(backingNum); ok {
				continue
			}
			if _, ok := seenNewBacking[backingNum]; ok {
				continue
			}
			seenNewBacking[backingNum] = struct{}{}
			ve.CreatedBackingTables = append(ve.CreatedBackingTables, e.source.TableBacking)
		}

		var metrics levelMetricsDelta
		for _, e := range entries {
			levelMetrics := metrics.level(e.level)
			levelMetrics.TablesIngested.Inc(e.virtual.Size)
		}

		return versionUpdate{
			VE:                      ve,
			JobID:                   jobID,
			Metrics:                 metrics,
			InProgressCompactionsFn: func() []compactionInfo { return d.getInProgressCompactionInfoLocked(nil) },
		}, nil
	})
	if err != nil {
		return false, err
	}
	if aborted {
		return true, nil
	}
	// Publish the new version through the read state so subsequent reads see
	// the cloned files.
	d.updateReadStateLocked(d.opts.DebugCheck)
	return false, nil
}
