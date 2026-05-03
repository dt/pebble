// Copyright 2026 The LevelDB-Go and Pebble Authors. All rights reserved. Use
// of this source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package pebble

import (
	"bytes"
	"context"

	"github.com/cockroachdb/errors"
	"github.com/cockroachdb/pebble/internal/base"
	"github.com/cockroachdb/pebble/internal/keyspan"
	"github.com/cockroachdb/pebble/internal/manifest"
	"github.com/cockroachdb/pebble/objstorage"
	"github.com/cockroachdb/pebble/sstable"
	"github.com/cockroachdb/pebble/sstable/blob"
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
//   - A source sstable contains a range deletion or range key whose
//     [Start, End) interval straddles a srcSpan boundary (partly inside,
//     partly outside).
//   - A straddling source sstable's in-span data blocks have a stored
//     block-shared prefix shorter than srcPrefix.
//   - All levels are saturated for a cloned file (extreme dst-conflict).
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
//
// Atomicity: a single VersionEdit installs all virtual SSTs and any small
// physical SSTs produced for boundary-block rewrites. On conflict with a
// concurrent compaction/excise on a referenced source SST, the operation
// restarts (bounded retry); orphaned boundary SSTs from the failed attempt
// are cleaned up.
//
// Level placement: cloned files are placed top-down, source-floored. Each
// cloned file is assigned to the deepest level >= its source level whose
// destination-space bounds don't overlap any existing file at that level.
// Source files at L0 may be placed at any L0 sublevel down to any deeper
// level. If even L0 is saturated for a particular cloned file (extreme dst
// conflict), the operation returns ErrUnsupportedClone.
//
// Returns ErrUnsupportedClone (with details) for v1 unsupported cases:
//   - rowblk-format SSTs intersecting srcSpan
//   - a source SST that contains a range deletion or range key whose
//     [Start, End) interval straddles a srcSpan boundary
//   - a straddling source SST whose in-span data blocks have a stored
//     shared prefix shorter than srcPrefix
//   - destination region is so saturated no level can host a cloned file
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
	dstSpan, err := computeDstSpan(srcSpan, srcPrefix, dstPrefix)
	if err != nil {
		return err
	}

	for attempt := 0; attempt < virtualCloneMaxRetries; attempt++ {
		retried, err := d.virtualCloneAttempt(ctx, attempt, srcSpan, srcPrefix, dstPrefix, dstSpan)
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

// computeDstSpan returns the dst-prefix-space span derived from srcSpan: the
// region of dst keys that VirtualClone owns and will excise atomically with
// the install of any cloned virtual/physical SSTs. The returned span is
// independent of which (if any) source data falls into srcSpan — the
// post-condition of VirtualClone is that the dst span is a snapshot of src,
// and that snapshot is empty when src is empty.
//
// srcSpan.Start is guaranteed by validateVirtualCloneInputs to start with
// srcPrefix, so dstSpan.Start is a straight bytewise translation. srcSpan.End
// has two cases:
//
//   - It starts with srcPrefix (e.g. srcSpan = [/t/1/k5, /t/1/k9)): translate
//     bytewise.
//   - It does not (e.g. srcSpan = [/t/1/, /t/2/)): the caller wants every
//     key with srcPrefix at or above srcSpan.Start cloned, so the
//     corresponding dst region is everything with dstPrefix at or above
//     dstSpan.Start. We close that with bytesPrefixEnd(dstPrefix), the
//     smallest byte string strictly greater than every byte string starting
//     with dstPrefix.
func computeDstSpan(srcSpan KeyRange, srcPrefix, dstPrefix []byte) (KeyRange, error) {
	start := translateUserKey(srcPrefix, dstPrefix, srcSpan.Start)
	var end []byte
	if bytes.HasPrefix(srcSpan.End, srcPrefix) {
		end = translateUserKey(srcPrefix, dstPrefix, srcSpan.End)
	} else {
		end = bytesPrefixEnd(dstPrefix)
		if end == nil {
			return KeyRange{}, errors.Newf(
				"pebble: VirtualClone cannot derive dst excise span: dstPrefix %q has no representable bytes-prefix successor",
				dstPrefix)
		}
	}
	return KeyRange{Start: start, End: end}, nil
}

// bytesPrefixEnd returns the smallest byte string strictly greater than every
// byte string having `prefix` as a bytes-prefix. It increments the last
// non-0xff byte and truncates trailing 0xff bytes. Returns nil if `prefix` is
// empty or consists entirely of 0xff bytes (no representable upper bound).
func bytesPrefixEnd(prefix []byte) []byte {
	end := append([]byte(nil), prefix...)
	for i := len(end) - 1; i >= 0; i-- {
		if end[i] < 0xff {
			end[i]++
			return end[:i+1]
		}
	}
	return nil
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
	// Correctness invariant for BlockPrefixSubstitution: srcPrefix is substituted
	// for dstPrefix as a literal byte-range replacement at the start of every
	// in-block key, so the byte at offset i in any translated key (for i >=
	// len(dstPrefix)) corresponds to the byte at offset i + len(srcPrefix) -
	// len(dstPrefix) in the original. Requiring equal-length prefixes makes that
	// shift zero, so every byte offset (and therefore every Split result, for
	// any tail-determined Comparer.Split) is preserved. This is the simplest
	// sufficient condition and covers the v1 caller (CockroachDB tenant clone
	// with same-length varint tenant IDs).
	//
	// We deliberately do NOT call Comparer.Split on srcPrefix or dstPrefix here:
	// Split is contractually defined on full encoded keys, and producing a
	// well-defined result for an arbitrary byte prefix is not required. CRDB's
	// Split, for example, reads a trailing length byte and returns nonsense for
	// inputs that aren't full keys. Earlier revisions of this check did probe
	// Split on the prefixes; that was both unsound (undefined behavior on
	// non-keys) and an unhelpful proxy (it rejected raw tenant prefixes that
	// are correct, while accepting "engine-encoded" prefixes that no real key
	// in the source span actually starts with). Don't reintroduce a Split-based
	// check here without first defining Split's contract on partial keys.
	if len(srcPrefix) != len(dstPrefix) {
		return errors.Newf(
			"pebble: VirtualClone srcPrefix and dstPrefix must have the same length "+
				"(len(srcPrefix)=%d, len(dstPrefix)=%d)",
			len(srcPrefix), len(dstPrefix))
	}
	if !bytes.HasPrefix(srcSpan.Start, srcPrefix) {
		return errors.Newf(
			"pebble: VirtualClone srcSpan start %q does not have srcPrefix %q",
			srcSpan.Start, srcPrefix)
	}
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

// clonePlanEntry describes one cloned file (virtual or physical) to install.
type clonePlanEntry struct {
	// sourceLevel is the level of the originating source SST (the floor for
	// destination-level placement).
	sourceLevel int
	// source is the source TableMetadata (only set for virtual entries; nil
	// for boundary-block physical entries).
	source *manifest.TableMetadata
	// virtual is the virtual TableMetadata to install. Mutually exclusive with
	// physical.
	virtual *manifest.TableMetadata
	// physical is the physical TableMetadata to install (boundary-block
	// rewrite). Mutually exclusive with virtual.
	physical *manifest.TableMetadata
	// assignedLevel is the level chosen by placeClonedFiles. -1 if not yet
	// assigned.
	assignedLevel int
}

// virtualCloneAttempt performs one attempt at VirtualClone.
func (d *DB) virtualCloneAttempt(
	ctx context.Context, attempt int, srcSpan KeyRange, srcPrefix, dstPrefix []byte, dstSpan KeyRange,
) (retried bool, _ error) {
	// Before snapshotting the version, check whether any memtable contains
	// keys overlapping srcSpan or dstSpan. If so, force a flush and wait for
	// it before proceeding. The src wait is required because recent writes to
	// keys in srcSpan that haven't yet flushed would otherwise be silently
	// absent from the cloned destination. The dst wait is required because
	// VirtualClone atomically excises dstSpan; unflushed writes to dstSpan
	// would silently survive the excise. This mirrors the pattern used by
	// DB.Compact (db.go:1810-1859) and the memtable-overlap handling in
	// DB.ingest (ingest.go:1863-1962).
	if retried, err := d.flushMemtablesOverlappingClone(ctx, srcSpan, dstSpan); err != nil || retried {
		return retried, err
	}

	d.mu.Lock()
	currentVersion := d.mu.versions.currentVersion()
	currentVersion.Ref()
	d.mu.Unlock()
	defer currentVersion.Unref()

	// Test hook: invoked after the version snapshot has been taken and d.mu
	// has been released. Tests use this to inject concurrent compactions or
	// excises into the race window between snapshot and apply.
	if hook := d.opts.private.testingCloneAfterSnapshot; hook != nil {
		hook(attempt)
	}

	srcSpanBounds := srcSpan.UserKeyBounds()

	// Shared blob value fetcher for the lifetime of this attempt. Used by
	// boundary-block reads (and any other block-level helper) to materialize
	// values whose handles point into blob files. Per-source TableBlobContexts
	// constructed downstream wire this fetcher together with the source
	// table's own BlobReferences.
	var blobFetcher blob.ValueFetcher
	blobFetcher.Init(&currentVersion.BlobFiles, d.fileCache,
		block.ReadEnv{}, blob.SuggestedCachedReaders(currentVersion.MaxReadAmp()))
	defer func() { _ = blobFetcher.Close() }()

	// Track physical SSTs we wrote pre-VE. On any failure path (retry,
	// terminal error) before the VE applies, we must remove them from the
	// object provider to avoid orphan files. Mirrors ingest.go's
	// ingestCleanup pattern (ingest.go:834).
	var preVEObjects []base.DiskFileNum
	cleanupPreVE := func() {
		for _, fn := range preVEObjects {
			_ = d.objProvider.Remove(base.FileTypeTable, fn)
		}
		preVEObjects = nil
	}

	var entries []clonePlanEntry
	// Walk LSM in source-level order (L0 sublevels first, then L1..L6),
	// collecting cloned entries.
	for layer, ls := range currentVersion.AllLevelsAndSublevels() {
		level := layer.Level()
		for m := range ls.Overlaps(d.cmp, srcSpanBounds).All() {
			// Validate that any range deletions and range keys in the source
			// table either lie entirely inside or entirely outside srcSpan.
			// Fragments that straddle a srcSpan boundary cannot be cleanly
			// substituted (we don't split them at the boundary in this slice)
			// and so we return ErrUnsupportedClone in that case. Also collect
			// the bounds of any in-span fragments so they can be reflected on
			// the cloned virtual SST's bounds.
			survey, err := d.surveyAndValidateFragments(ctx, m, srcSpan, level)
			if err != nil {
				cleanupPreVE()
				return false, err
			}
			fullyContained := srcSpanBounds.ContainsInternalKey(d.cmp, m.Smallest()) &&
				srcSpanBounds.ContainsInternalKey(d.cmp, m.Largest())
			if fullyContained {
				if !bytes.HasPrefix(m.Smallest().UserKey, srcPrefix) ||
					!bytes.HasPrefix(m.Largest().UserKey, srcPrefix) {
					cleanupPreVE()
					return false, errors.Wrapf(ErrUnsupportedClone,
						"source table %s at L%d has bounds outside srcPrefix %q (smallest=%s largest=%s)",
						m.TableNum, level, srcPrefix,
						m.Smallest().Pretty(d.opts.Comparer.FormatKey),
						m.Largest().Pretty(d.opts.Comparer.FormatKey))
				}
				vm, err := d.buildFullyContainedVirtual(m, srcPrefix, dstPrefix)
				if err != nil {
					cleanupPreVE()
					return false, err
				}
				entries = append(entries, clonePlanEntry{
					sourceLevel:   level,
					source:        m,
					virtual:       vm,
					assignedLevel: -1,
				})
				continue
			}
			// Straddler: open the source, read its index, classify blocks,
			// build a block-aligned virtual TableMetadata + 0-2 boundary
			// physical SSTs.
			straddlerEntries, written, err := d.buildStraddlerEntries(
				ctx, m, level, srcSpan, srcSpanBounds, srcPrefix, dstPrefix, survey, &blobFetcher)
			if err != nil {
				// Track any objects already written before propagating.
				preVEObjects = append(preVEObjects, written...)
				cleanupPreVE()
				return false, err
			}
			preVEObjects = append(preVEObjects, written...)
			entries = append(entries, straddlerEntries...)
		}
	}

	dstSpanBounds := dstSpan.UserKeyBounds()

	// Source/destination keyspan disjointness: VirtualClone owns dstSpan and
	// will excise it. If a source SST's bounds extend into dstSpan, the same
	// physical file would be touched by both the source-stand-in path and the
	// dst-excise path (producing conflicting DeletedTables entries). This is a
	// pathological layout (a single SST spanning both src-prefix and dst-prefix
	// keys); reject it as ErrUnsupportedClone.
	for _, e := range entries {
		if e.source == nil {
			continue
		}
		if dstSpanBounds.Overlaps(d.cmp, e.source.UserKeyBounds()) {
			cleanupPreVE()
			return false, errors.Wrapf(ErrUnsupportedClone,
				"source table %s at L%d has bounds overlapping the dst excise span (smallest=%s largest=%s)",
				e.source.TableNum, e.sourceLevel,
				e.source.Smallest().Pretty(d.opts.Comparer.FormatKey),
				e.source.Largest().Pretty(d.opts.Comparer.FormatKey))
		}
	}

	// Each cloned entry is placed at its source level. After dst excise,
	// dstSpan is empty at every level, so cloned-file bounds (which lie within
	// dstSpan) cannot overlap any surviving file at any level. Within a single
	// source level, two cloned files originate from non-overlapping source
	// SSTs (LSM invariant) and so their dst-translated bounds also don't
	// overlap.
	for i := range entries {
		entries[i].assignedLevel = entries[i].sourceLevel
	}

	// Test hook: invoked just before re-acquiring d.mu and applying the
	// version edit.
	if hook := d.opts.private.testingCloneBeforeUpdateVersionLocked; hook != nil {
		hook(attempt)
	}

	// Apply via UpdateVersionLocked.
	d.mu.Lock()
	jobID := d.newJobIDLocked()
	defer d.mu.Unlock()

	var aborted bool
	_, err := d.mu.versions.UpdateVersionLocked(func() (versionUpdate, error) {
		current := d.mu.versions.currentVersion()

		// Re-validate every source backing still exists at the level we
		// observed during the read phase. If anything has shifted, abort.
		for _, e := range entries {
			if e.source == nil {
				continue
			}
			if !current.Contains(e.sourceLevel, e.source) {
				aborted = true
				return versionUpdate{}, nil
			}
		}

		ve := &manifest.VersionEdit{}
		ve.DeletedTables = make(map[manifest.DeletedTableEntry]*manifest.TableMetadata)
		// Per-source bookkeeping. We must visit each unique source SST at
		// most once for the stand-in / backing-promotion bookkeeping below,
		// even when it produces multiple cloned entries (e.g. a straddler
		// with both a virtual in-span run and one or two boundary
		// physicals).
		seenSource := make(map[base.TableNum]struct{})
		seenNewBacking := make(map[base.DiskFileNum]struct{})
		for _, e := range entries {
			meta := e.virtual
			if meta == nil {
				meta = e.physical
			}
			ve.NewTables = append(ve.NewTables, manifest.NewTableEntry{
				Level: e.assignedLevel,
				Meta:  meta,
			})
			if e.source == nil {
				// Boundary-rewrite physical SSTs are self-contained: they
				// have a fresh backing initialized by InitPhysicalBacking and
				// no source-backing sharing to arrange.
				continue
			}
			if _, ok := seenSource[e.source.TableNum]; ok {
				continue
			}
			seenSource[e.source.TableNum] = struct{}{}

			// Source-backing sharing. The cloned virtual SST(s) we emit
			// AttachVirtualBacking(source.TableBacking), so the manifest must
			// see this backing as a virtual backing (in latest.virtualBackings).
			//
			// If the source is itself virtual, its backing is already in the
			// virtual-backings registry — nothing to do.
			//
			// If the source is physical, the backing is currently tracked
			// only by the source's own TableMetadata. Promoting it to a
			// virtual backing while the source physical SST remains live
			// would double-track the backing: a future compaction that
			// deletes the source would mark the backing as a physical zombie
			// in getZombieTablesAndUpdateVirtualBackings (since the backing
			// is in DeletedTables and not in that compaction's stillUsed
			// set), but virtualBackings.AddAndRef has already taken a long-
			// lived reference that no later VE can release until the cloned
			// virtual is itself removed. The backing sits in zombieTables
			// with a non-zero refcount forever, fataling Close with
			// "non-zero zombie file count".
			//
			// To avoid that, replace the source physical SST with a virtual
			// stand-in covering its identical bounds in this same VE. The
			// stand-in references the same backing via AttachVirtualBacking
			// (no data movement, no rewrite); from the LSM's perspective the
			// source's data is unchanged in src space. The backing is now
			// referenced only via virtualBackings, satisfying the manifest's
			// "either physical or virtual, not both" invariant. This mirrors
			// the excise.go pattern for shrinking a physical SST.
			if !e.source.Virtual {
				standIn, err := buildSourceStandIn(d.cmp, e.source, d.mu.versions.getNextTableNum(),
					d.opts.Comparer.FormatKey, d.FormatMajorVersion())
				if err != nil {
					return versionUpdate{}, err
				}
				ve.DeletedTables[manifest.DeletedTableEntry{
					Level:   e.sourceLevel,
					FileNum: e.source.TableNum,
				}] = e.source
				ve.NewTables = append(ve.NewTables, manifest.NewTableEntry{
					Level: e.sourceLevel,
					Meta:  standIn,
				})
			}

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

		// Excise dstSpan against the live version. VirtualClone's contract
		// is that dst becomes a snapshot of src, so any pre-existing data in
		// dstSpan must be removed atomically with the install of any cloned
		// virtual/physical SSTs. We compute the deletion set from `current`
		// (not the read-phase snapshot) so we capture files added by
		// concurrent flushes/compactions between our snapshot and the apply.
		//
		// Source files (which we replace with a stand-in in this same VE)
		// have srcPrefix bounds and so cannot overlap dstSpan; the
		// pre-flight check above already rejected the pathological case of
		// a single source SST whose bounds extend into dstSpan.
		//
		// TODO: thread an excise sequence number allocated via the commit
		// pipeline so concurrent EFOS protected ranges over dstSpan are
		// honored. For now we record the current logSeqNum as a best-effort
		// marker; the dst-tenant-clone use case (which assumes dst is empty
		// and not under concurrent access) does not exercise EFOS overlap.
		exciseSeqNum := d.mu.versions.logSeqNum.Load()
		var anyExcised bool
		for layer, ls := range current.AllLevelsAndSublevels() {
			for m := range ls.Overlaps(d.cmp, dstSpanBounds).All() {
				leftTable, rightTable, err := d.exciseTable(
					ctx, dstSpanBounds, m, layer.Level(), tightExciseBoundsIfLocal)
				if err != nil {
					return versionUpdate{}, err
				}
				applyExciseToVersionEdit(ve, m, leftTable, rightTable, layer.Level())
				anyExcised = true
			}
		}
		if anyExcised {
			ve.ExciseBoundsRecord = append(ve.ExciseBoundsRecord, manifest.ExciseOpEntry{
				Bounds: dstSpanBounds,
				SeqNum: exciseSeqNum,
			})
			// Cancel any in-progress compaction whose output bounds overlap
			// dstSpan: the compaction may write a file landing in dstSpan
			// after the excise applies, defeating it. Mirrors the pattern in
			// ingest.go:2457-2487.
			for c := range d.mu.compact.inProgress {
				if c.VersionEditApplied() {
					continue
				}
				bounds := c.Bounds()
				if bounds != nil && bounds.Overlaps(d.cmp, dstSpanBounds) {
					c.Cancel()
				}
			}
		}

		var metrics levelMetricsDelta
		for _, e := range entries {
			meta := e.virtual
			if meta == nil {
				meta = e.physical
			}
			levelMetrics := metrics.level(e.assignedLevel)
			levelMetrics.TablesIngested.Inc(meta.Size)
		}

		// If there is nothing to clone and nothing to excise, skip the VE.
		if len(entries) == 0 && !anyExcised {
			return versionUpdate{}, nil
		}

		return versionUpdate{
			VE:                      ve,
			JobID:                   jobID,
			Metrics:                 metrics,
			InProgressCompactionsFn: func() []compactionInfo { return d.getInProgressCompactionInfoLocked(nil) },
		}, nil
	})
	if err != nil {
		cleanupPreVE()
		return false, err
	}
	if aborted {
		cleanupPreVE()
		return true, nil
	}
	d.updateReadStateLocked(d.opts.DebugCheck)
	return false, nil
}

// flushMemtablesOverlappingClone walks d.mu.mem.queue and forces a flush of
// the newest memtable whose contents overlap srcSpan or dstSpan, waiting for
// the flush to complete before returning. The returned retried bool is true
// when a flush was forced (signalling the caller to start a fresh attempt
// with a fresh version snapshot).
//
// This mirrors the memtable-overlap pattern in DB.Compact (db.go:1810-1859):
// walk the queue from newest to oldest, find the newest overlapping
// memtable, force a rotation if it's the mutable one, schedule a flush, and
// then wait on the flushed channel.
func (d *DB) flushMemtablesOverlappingClone(
	ctx context.Context, srcSpan, dstSpan KeyRange,
) (retried bool, _ error) {
	d.mu.Lock()
	mem, err := func() (*flushableEntry, error) {
		// Walk from newest (mutable) to oldest. We only need to wait on the
		// newest overlapping memtable; once it (and any older ones already
		// scheduled to flush) finishes, all of those keys are in L0.
		for i := len(d.mu.mem.queue) - 1; i >= 0; i-- {
			mem := d.mu.mem.queue[i]
			var anyOverlaps bool
			mem.computePossibleOverlaps(func(b bounded) shouldContinue {
				anyOverlaps = true
				return stopIteration
			}, srcSpan, dstSpan)
			if !anyOverlaps {
				continue
			}
			var err error
			if mem.flushable == d.mu.mem.mutable {
				// We must hold both commitPipeline.mu and DB.mu when calling
				// makeRoomForWrite. Lock order forces us to release DB.mu so
				// we can grab commit.mu first.
				d.mu.Unlock()
				d.commit.mu.Lock()
				d.mu.Lock()
				defer d.commit.mu.Unlock() //nolint:deferloop
				if mem.flushable == d.mu.mem.mutable {
					err = d.makeRoomForWrite(nil)
				}
			}
			mem.flushForced = true
			d.maybeScheduleFlush()
			return mem, err
		}
		return nil, nil
	}()
	d.mu.Unlock()

	if err != nil {
		return false, err
	}
	if mem == nil {
		return false, nil
	}
	select {
	case <-mem.flushed:
	case <-ctx.Done():
		return false, ctx.Err()
	}
	return true, nil
}

// buildSourceStandIn produces a virtual TableMetadata that covers the full
// bounds of a physical source SST. It is used to replace the source physical
// SST in the same VE that promotes the source's backing to a virtual backing
// for the cloned virtual SST(s) to share. The stand-in carries no
// substitution: it surfaces the source's data unchanged in src space.
//
// The stand-in shares the source's TableBacking via AttachVirtualBacking and
// inherits the source's bounds, blob references, blob-reference depth, and
// any synthetic prefix/suffix transforms. No data is read or written.
func buildSourceStandIn(
	cmp base.Compare,
	src *manifest.TableMetadata,
	tableNum base.TableNum,
	formatKey base.FormatKey,
	fmv FormatMajorVersion,
) (*manifest.TableMetadata, error) {
	standIn := &manifest.TableMetadata{
		Virtual:                  true,
		TableNum:                 tableNum,
		SeqNums:                  src.SeqNums,
		LargestSeqNumAbsolute:    src.LargestSeqNumAbsolute,
		SyntheticPrefixAndSuffix: src.SyntheticPrefixAndSuffix,
		BlobReferenceDepth:       src.BlobReferenceDepth,
	}
	if src.HasPointKeys {
		standIn.ExtendPointKeyBounds(cmp, src.PointKeyBounds.Smallest(), src.PointKeyBounds.Largest())
	}
	if src.HasRangeKeys {
		standIn.ExtendRangeKeyBounds(cmp, src.RangeKeyKinds, src.RangeKeyBounds.Smallest(), src.RangeKeyBounds.Largest())
	}
	standIn.AttachVirtualBacking(src.TableBacking)
	standIn.Size = src.Size
	if standIn.Size == 0 {
		standIn.Size = 1
	}
	determineExcisedTableBlobReferences(src.BlobReferences, src.Size, standIn, fmv)
	if err := standIn.Validate(cmp, formatKey); err != nil {
		return nil, errors.Wrapf(err,
			"pebble: VirtualClone produced invalid source stand-in for %s", src.TableNum)
	}
	standIn.ValidateVirtual(src)
	return standIn, nil
}

// buildFullyContainedVirtual produces a virtual TableMetadata for a source
// table whose bounds lie entirely within srcSpan.
func (d *DB) buildFullyContainedVirtual(
	m *manifest.TableMetadata, srcPrefix, dstPrefix []byte,
) (*manifest.TableMetadata, error) {
	// Reject row-based table format (no per-block stored shared prefix).
	var isRowblk bool
	if err := d.fileCache.withReader(context.TODO(), block.NoReadEnv, m,
		func(r *sstable.Reader, _ sstable.ReadEnv) error {
			format, err := r.TableFormat()
			if err != nil {
				return err
			}
			isRowblk = !format.BlockColumnar()
			return nil
		}); err != nil {
		return nil, errors.Wrapf(err,
			"pebble: VirtualClone failed to read format of source table %s", m.TableNum)
	}
	if isRowblk {
		return nil, errors.Wrapf(ErrUnsupportedClone,
			"source table %s uses the row-based block format", m.TableNum)
	}

	if !m.HasPointKeys {
		return nil, errors.AssertionFailedf(
			"pebble: VirtualClone source table %s has no point keys", m.TableNum)
	}

	vm := &manifest.TableMetadata{
		Virtual:               true,
		TableNum:              d.mu.versions.getNextTableNum(),
		SeqNums:               m.SeqNums,
		LargestSeqNumAbsolute: m.LargestSeqNumAbsolute,
		BlobReferenceDepth:    m.BlobReferenceDepth,
		BlockPrefixSubstitution: sstable.BlockPrefixSubstitution{
			Src: append([]byte(nil), srcPrefix...),
			Dst: append([]byte(nil), dstPrefix...),
		},
	}
	smallest := translateInternalKey(srcPrefix, dstPrefix, m.PointKeyBounds.Smallest())
	largest := translateInternalKey(srcPrefix, dstPrefix, m.PointKeyBounds.Largest())
	vm.ExtendPointKeyBounds(d.cmp, smallest, largest)
	if m.HasRangeKeys {
		rkSmallest := translateInternalKey(srcPrefix, dstPrefix, m.RangeKeyBounds.Smallest())
		rkLargest := translateInternalKey(srcPrefix, dstPrefix, m.RangeKeyBounds.Largest())
		vm.ExtendRangeKeyBounds(d.cmp, m.RangeKeyKinds, rkSmallest, rkLargest)
	}

	vm.AttachVirtualBacking(m.TableBacking)
	vm.Size = m.Size
	if vm.Size == 0 {
		vm.Size = 1
	}
	determineExcisedTableBlobReferences(m.BlobReferences, m.Size, vm, d.FormatMajorVersion())

	if err := vm.Validate(d.cmp, d.opts.Comparer.FormatKey); err != nil {
		return nil, errors.Wrapf(err,
			"pebble: VirtualClone produced invalid virtual table for source %s", m.TableNum)
	}
	vm.ValidateVirtual(m)
	return vm, nil
}

// buildStraddlerEntries handles a single straddling source SST. It reads the
// SST's index, identifies the contiguous run of in-span data blocks and the
// at-most-two boundary blocks, validates the per-block precondition, builds a
// block-aligned virtual TableMetadata, and writes one physical SST per
// non-empty boundary block.
//
// On error, any objects that have already been written to the object provider
// are returned in `written` so the caller can clean them up.
func (d *DB) buildStraddlerEntries(
	ctx context.Context,
	m *manifest.TableMetadata,
	level int,
	srcSpan KeyRange,
	srcSpanBounds base.UserKeyBounds,
	srcPrefix, dstPrefix []byte,
	survey fragmentSurvey,
	blobFetcher *blob.ValueFetcher,
) (entries []clonePlanEntry, written []base.DiskFileNum, _ error) {
	if !m.HasPointKeys {
		return nil, nil, errors.AssertionFailedf(
			"pebble: VirtualClone source table %s has no point keys", m.TableNum)
	}

	// Per-source-table blob context. The fetcher is shared across all source
	// SSTs in this attempt; the References are this source SST's own blob
	// references (used to map a row's reference index to a blob file ID).
	blobContext := sstable.TableBlobContext{
		ValueFetcher: blobFetcher,
		References:   &m.BlobReferences,
	}

	var blocks []blockInfo

	var isRowblk bool

	// Note: WalkDataBlocks transparently handles both single-level and
	// two-level index layouts (top-level -> second-level -> data block).
	// The boundary classification below operates on the resulting flat list
	// of (separator, handle) pairs and is independent of index depth, as is
	// the per-block precondition validation and boundary-block rewrite.
	err := d.fileCache.withReader(ctx, block.NoReadEnv, m,
		func(r *sstable.Reader, _ sstable.ReadEnv) error {
			format, err := r.TableFormat()
			if err != nil {
				return err
			}
			if !format.BlockColumnar() {
				isRowblk = true
				return nil
			}
			return r.WalkDataBlocks(ctx, func(e sstable.DataBlockEntry) error {
				blocks = append(blocks, blockInfo{
					separator: append([]byte(nil), e.Separator...),
					handle:    e.Handle,
				})
				return nil
			})
		})
	if err != nil {
		return nil, nil, errors.Wrapf(err,
			"pebble: VirtualClone failed to read index of source table %s", m.TableNum)
	}
	if isRowblk {
		return nil, nil, errors.Wrapf(ErrUnsupportedClone,
			"source table %s at L%d uses the row-based block format", m.TableNum, level)
	}
	if len(blocks) == 0 {
		return nil, nil, errors.AssertionFailedf(
			"pebble: VirtualClone source table %s has no data blocks", m.TableNum)
	}

	// Classify each block by its key range relative to srcSpan. A block's
	// upper bound (inclusive) is its separator. A block's lower bound is the
	// previous block's separator (exclusive), or the table's smallest user
	// key (inclusive) for block 0.
	//
	// in-span: lower >= srcSpan.Start AND separator < srcSpan.End
	// boundary-lo: separator >= srcSpan.Start AND lower < srcSpan.Start
	// boundary-hi: lower < srcSpan.End AND separator >= srcSpan.End
	// outside: separator < srcSpan.Start OR lower >= srcSpan.End

	cmp := d.cmp
	// firstInSpan and lastInSpan delimit the contiguous in-span run (closed
	// interval); -1 means no in-span block.
	firstInSpan, lastInSpan := -1, -1
	// boundaryLo and boundaryHi are the 0-or-1 boundary block indices.
	boundaryLo, boundaryHi := -1, -1

	tableSmallest := m.Smallest().UserKey
	for i, blk := range blocks {
		var lower []byte
		if i == 0 {
			lower = tableSmallest
		} else {
			// Lower exclusive bound = previous block's separator. We treat it
			// inclusively for classification: a block whose first key equals
			// the previous separator can still happen in principle. We use the
			// previous separator as a lower-bound proxy.
			lower = blocks[i-1].separator
		}
		sep := blk.separator

		// Block is fully outside srcSpan if separator < srcSpan.Start (entirely
		// before) or lower >= srcSpan.End (entirely after).
		if cmp(sep, srcSpan.Start) < 0 {
			continue
		}
		if cmp(lower, srcSpan.End) >= 0 {
			continue
		}
		// Block at least partially intersects srcSpan.
		lowerInSpan := cmp(lower, srcSpan.Start) >= 0
		// For "fully in-span" we need separator strictly less than srcSpan.End.
		// (separator equals last-key, last key < srcSpan.End requires sep < End.)
		upperInSpan := cmp(sep, srcSpan.End) < 0
		switch {
		case lowerInSpan && upperInSpan:
			if firstInSpan == -1 {
				firstInSpan = i
			}
			lastInSpan = i
		case !lowerInSpan && upperInSpan:
			// Boundary-lo block.
			if boundaryLo != -1 {
				return nil, nil, errors.AssertionFailedf(
					"pebble: VirtualClone source %s has multiple lo-boundary blocks", m.TableNum)
			}
			boundaryLo = i
		case lowerInSpan && !upperInSpan:
			// Boundary-hi block.
			if boundaryHi != -1 {
				return nil, nil, errors.AssertionFailedf(
					"pebble: VirtualClone source %s has multiple hi-boundary blocks", m.TableNum)
			}
			boundaryHi = i
		case !lowerInSpan && !upperInSpan:
			// Single block straddling both ends — use it as both.
			if boundaryLo != -1 || boundaryHi != -1 {
				return nil, nil, errors.AssertionFailedf(
					"pebble: VirtualClone source %s has both boundaries in a single block but other boundaries already set",
					m.TableNum)
			}
			boundaryLo = i
			// The same block carries both boundaries; we'll iterate it once.
			boundaryHi = -1
		}
	}

	// Build virtual entry for the in-span run, if any.
	if firstInSpan >= 0 {
		// Per-block precondition validation: every in-span block's stored
		// shared prefix must start with srcPrefix.
		if err := d.validateInSpanBlocks(ctx, m, blocks[firstInSpan:lastInSpan+1], srcPrefix, blobContext); err != nil {
			return nil, nil, err
		}

		// Block-aligned bounds in dst-space. The virtual SST covers blocks
		// [firstInSpan..lastInSpan]. Its smallest is the lower bound of
		// block firstInSpan; its largest is the separator of lastInSpan.
		// We use the previous block's separator (or tableSmallest for the
		// first block) as the smallest bound. If we use a separator that
		// belongs to a boundary block, we need to be careful that the bound
		// actually skips the boundary block. The previous separator is <
		// any key in firstInSpan, so smallest := first key of block
		// firstInSpan. Since iteration uses these as user-key bounds, using
		// blocks[firstInSpan-1].separator as exclusive smallest could
		// include boundary keys. Safer: use the first key of block
		// firstInSpan itself.
		//
		// We materialize the smallest/largest by reading the data blocks at
		// the run's endpoints (we will read these blocks once anyway for
		// validation). To minimize complexity, we re-derive smallest =
		// firstKeyOf(blocks[firstInSpan]) and largest =
		// lastKeyOf(blocks[lastInSpan]). validateInSpanBlocks already opens
		// these blocks; we extract while we're there.
		firstIK, lastIK, err := d.firstAndLastKeyOfRun(ctx, m, blocks[firstInSpan], blocks[lastInSpan], blobContext)
		if err != nil {
			return nil, nil, err
		}
		// Translate into dst-space, preserving original trailers.
		smallest := translateInternalKey(srcPrefix, dstPrefix, firstIK)
		largest := translateInternalKey(srcPrefix, dstPrefix, lastIK)

		vm := &manifest.TableMetadata{
			Virtual:               true,
			TableNum:              d.mu.versions.getNextTableNum(),
			SeqNums:               m.SeqNums,
			LargestSeqNumAbsolute: m.LargestSeqNumAbsolute,
			BlobReferenceDepth:    m.BlobReferenceDepth,
			BlockPrefixSubstitution: sstable.BlockPrefixSubstitution{
				Src: append([]byte(nil), srcPrefix...),
				Dst: append([]byte(nil), dstPrefix...),
			},
		}
		vm.ExtendPointKeyBounds(d.cmp, smallest, largest)
		// Extend the virtual SST's point-key bounds to include any in-span
		// range deletions (which are tracked under point keys), and set its
		// range-key bounds for any in-span range keys. The fragments are
		// surfaced via NewRawRangeDelIter / NewRawRangeKeyIter on the virtual
		// reader, with bounds-truncation against the virtual SST's bounds; so
		// the bounds must encompass every fragment that should be visible.
		if survey.hasInSpanRangeDel {
			vm.ExtendPointKeyBounds(d.cmp,
				translateInternalKey(srcPrefix, dstPrefix, survey.smallestRangeDel),
				translateInternalKey(srcPrefix, dstPrefix, survey.largestRangeDel))
		}
		if survey.hasInSpanRangeKey {
			vm.ExtendRangeKeyBounds(d.cmp, survey.rangeKeyKinds,
				translateInternalKey(srcPrefix, dstPrefix, survey.smallestRangeKey),
				translateInternalKey(srcPrefix, dstPrefix, survey.largestRangeKey))
		}

		vm.AttachVirtualBacking(m.TableBacking)
		// Approximate size by proportion of in-span blocks to total blocks.
		approx := uint64(0)
		for _, b := range blocks[firstInSpan : lastInSpan+1] {
			approx += b.handle.Length
		}
		vm.Size = approx
		if vm.Size == 0 {
			vm.Size = 1
		}
		determineExcisedTableBlobReferences(m.BlobReferences, m.Size, vm, d.FormatMajorVersion())

		if err := vm.Validate(d.cmp, d.opts.Comparer.FormatKey); err != nil {
			return nil, nil, errors.Wrapf(err,
				"pebble: VirtualClone produced invalid virtual table for source %s", m.TableNum)
		}
		vm.ValidateVirtual(m)

		entries = append(entries, clonePlanEntry{
			sourceLevel:   level,
			source:        m,
			virtual:       vm,
			assignedLevel: -1,
		})
	} else if survey.hasInSpanRangeDel || survey.hasInSpanRangeKey {
		// No in-span data block run, but the source has range deletions or
		// range keys whose bounds lie within srcSpan. Build a virtual SST
		// whose bounds cover only those fragments (in dst space). Point keys
		// from the boundary blocks (if any) are exposed via the boundary
		// physical SSTs; this virtual entry only surfaces fragments.
		vm := &manifest.TableMetadata{
			Virtual:               true,
			TableNum:              d.mu.versions.getNextTableNum(),
			SeqNums:               m.SeqNums,
			LargestSeqNumAbsolute: m.LargestSeqNumAbsolute,
			BlobReferenceDepth:    m.BlobReferenceDepth,
			BlockPrefixSubstitution: sstable.BlockPrefixSubstitution{
				Src: append([]byte(nil), srcPrefix...),
				Dst: append([]byte(nil), dstPrefix...),
			},
		}
		if survey.hasInSpanRangeDel {
			vm.ExtendPointKeyBounds(d.cmp,
				translateInternalKey(srcPrefix, dstPrefix, survey.smallestRangeDel),
				translateInternalKey(srcPrefix, dstPrefix, survey.largestRangeDel))
		}
		if survey.hasInSpanRangeKey {
			vm.ExtendRangeKeyBounds(d.cmp, survey.rangeKeyKinds,
				translateInternalKey(srcPrefix, dstPrefix, survey.smallestRangeKey),
				translateInternalKey(srcPrefix, dstPrefix, survey.largestRangeKey))
		}
		vm.AttachVirtualBacking(m.TableBacking)
		vm.Size = 1
		determineExcisedTableBlobReferences(m.BlobReferences, m.Size, vm, d.FormatMajorVersion())
		if err := vm.Validate(d.cmp, d.opts.Comparer.FormatKey); err != nil {
			return nil, nil, errors.Wrapf(err,
				"pebble: VirtualClone produced invalid virtual table for source %s", m.TableNum)
		}
		vm.ValidateVirtual(m)
		entries = append(entries, clonePlanEntry{
			sourceLevel:   level,
			source:        m,
			virtual:       vm,
			assignedLevel: -1,
		})
	}

	// Boundary-block rewrites.
	maybeRewrite := func(idx int) error {
		if idx < 0 {
			return nil
		}
		// Skip if this boundary block is the same as the in-span run (which
		// shouldn't happen by construction, but be safe).
		if idx >= firstInSpan && idx <= lastInSpan && firstInSpan >= 0 {
			return nil
		}
		physical, fileNum, err := d.rewriteBoundaryBlock(
			ctx, m, blocks[idx].handle.Handle, srcSpan, srcSpanBounds,
			srcPrefix, dstPrefix, level, blobContext)
		if err != nil {
			return err
		}
		if physical == nil {
			// Empty boundary (no in-span keys).
			return nil
		}
		written = append(written, fileNum)
		entries = append(entries, clonePlanEntry{
			sourceLevel:   level,
			physical:      physical,
			assignedLevel: -1,
		})
		return nil
	}

	if err := maybeRewrite(boundaryLo); err != nil {
		return entries, written, err
	}
	if err := maybeRewrite(boundaryHi); err != nil {
		return entries, written, err
	}
	return entries, written, nil
}

// blockInfo is a small (separator, handle) pair used in straddler analysis.
type blockInfo struct {
	separator []byte
	handle    block.HandleWithProperties
}

// fragmentSurvey summarizes the result of walking a source table's range
// deletion and range key blocks to classify each fragment relative to srcSpan.
//
// SrcSpace bounds (smallestRangeDel/largestRangeDel/etc.) are in the source
// table's storage prefix space; the caller is responsible for translating them
// to dst-prefix space before installing them on a cloned virtual TableMetadata.
type fragmentSurvey struct {
	// hasInSpanRangeDel is set if at least one range deletion fragment lies
	// entirely within srcSpan.
	hasInSpanRangeDel bool
	// smallestRangeDel/largestRangeDel are the bounds of the in-span range
	// deletion fragments, expressed as InternalKeys with appropriate trailers
	// (largest uses MakeExclusiveSentinelKey for the end). Valid only when
	// hasInSpanRangeDel is true.
	smallestRangeDel base.InternalKey
	largestRangeDel  base.InternalKey

	// hasInSpanRangeKey is set if at least one range key fragment lies entirely
	// within srcSpan.
	hasInSpanRangeKey bool
	smallestRangeKey  base.InternalKey
	largestRangeKey   base.InternalKey
	// rangeKeyKinds tracks the union of range key kinds observed among in-span
	// fragments.
	rangeKeyKinds manifest.RangeKeyKinds
}

// surveyAndValidateFragments walks the source table's range deletion and range
// key blocks (if present), verifies that every fragment's [Start, End)
// interval lies either entirely inside or entirely outside srcSpan, and
// returns a fragmentSurvey describing the in-span fragments.
//
// A straddling fragment (partly inside, partly outside srcSpan) returns
// ErrUnsupportedClone with details identifying the offending fragment.
//
// Fragments that lie entirely outside srcSpan are harmless: they will be
// excluded by the cloned virtual SST's bounds. Fragments fully inside
// srcSpan are translated at iteration time via the substitution-aware
// fragment iterator.
func (d *DB) surveyAndValidateFragments(
	ctx context.Context, m *manifest.TableMetadata, srcSpan KeyRange, level int,
) (fragmentSurvey, error) {
	cmp := d.cmp
	var survey fragmentSurvey
	walk := func(
		kind string,
		iter keyspan.FragmentIterator,
		recordInSpan func(s *keyspan.Span),
	) error {
		if iter == nil {
			return nil
		}
		defer iter.Close()
		for s, err := iter.First(); s != nil || err != nil; s, err = iter.Next() {
			if err != nil {
				return errors.Wrapf(err,
					"pebble: VirtualClone failed iterating %s of source table %s",
					kind, m.TableNum)
			}
			// Classify the fragment relative to srcSpan = [Start, End).
			//   fully-outside: s.End <= srcSpan.Start  OR  s.Start >= srcSpan.End
			//   fully-inside:  s.Start >= srcSpan.Start AND s.End <= srcSpan.End
			//   straddle:      everything else
			endLEStart := cmp(s.End, srcSpan.Start) <= 0
			startGEEnd := cmp(s.Start, srcSpan.End) >= 0
			if endLEStart || startGEEnd {
				continue
			}
			startInside := cmp(s.Start, srcSpan.Start) >= 0
			endInside := cmp(s.End, srcSpan.End) <= 0
			if startInside && endInside {
				recordInSpan(s)
				continue
			}
			return errors.Wrapf(ErrUnsupportedClone,
				"source table %s at L%d contains a %s fragment [%s, %s) that straddles srcSpan [%s, %s)",
				m.TableNum, level, kind,
				d.opts.Comparer.FormatKey(s.Start), d.opts.Comparer.FormatKey(s.End),
				d.opts.Comparer.FormatKey(srcSpan.Start), d.opts.Comparer.FormatKey(srcSpan.End))
		}
		return nil
	}

	err := d.fileCache.withReader(ctx, block.NoReadEnv, m,
		func(r *sstable.Reader, _ sstable.ReadEnv) error {
			rdel, err := r.NewRawRangeDelIter(ctx, sstable.NoFragmentTransforms, sstable.NoReadEnv)
			if err != nil {
				return errors.Wrapf(err,
					"pebble: VirtualClone failed opening range-del iterator on source %s",
					m.TableNum)
			}
			if err := walk("range deletion", rdel, func(s *keyspan.Span) {
				smallest := s.SmallestKey()
				largest := base.MakeExclusiveSentinelKey(base.InternalKeyKindRangeDelete, s.End)
				if !survey.hasInSpanRangeDel {
					survey.hasInSpanRangeDel = true
					survey.smallestRangeDel = smallest.Clone()
					survey.largestRangeDel = largest.Clone()
					return
				}
				if base.InternalCompare(cmp, smallest, survey.smallestRangeDel) < 0 {
					survey.smallestRangeDel = smallest.Clone()
				}
				if base.InternalCompare(cmp, largest, survey.largestRangeDel) > 0 {
					survey.largestRangeDel = largest.Clone()
				}
			}); err != nil {
				return err
			}

			if !m.HasRangeKeys {
				return nil
			}
			rkey, err := r.NewRawRangeKeyIter(ctx, sstable.NoFragmentTransforms, sstable.NoReadEnv)
			if err != nil {
				return errors.Wrapf(err,
					"pebble: VirtualClone failed opening range-key iterator on source %s",
					m.TableNum)
			}
			return walk("range key", rkey, func(s *keyspan.Span) {
				smallest := s.SmallestKey()
				// For range keys, the largest is an exclusive sentinel using
				// the maximum range-key kind. Mirroring keyspan.Span.LargestKey
				// behavior for the table's bounds.
				largest := base.MakeExclusiveSentinelKey(base.InternalKeyKindRangeKeyMax, s.End)
				if !survey.hasInSpanRangeKey {
					survey.hasInSpanRangeKey = true
					survey.smallestRangeKey = smallest.Clone()
					survey.largestRangeKey = largest.Clone()
				} else {
					if base.InternalCompare(cmp, smallest, survey.smallestRangeKey) < 0 {
						survey.smallestRangeKey = smallest.Clone()
					}
					if base.InternalCompare(cmp, largest, survey.largestRangeKey) > 0 {
						survey.largestRangeKey = largest.Clone()
					}
				}
				// We don't have a per-fragment kind classifier here; record
				// AnyRangeKeys defensively (matches the convention in
				// excise.go and other callers without precise information).
				survey.rangeKeyKinds = manifest.AnyRangeKeys
			})
		})
	return survey, err
}

// validateInSpanBlocks reads each of the provided blocks and verifies that
// every key in the block starts with srcPrefix. For blocks fully within
// srcSpan ⊆ [srcPrefix, srcPrefix.next), this holds by construction; the
// check is defense in depth against pathological key layouts or comparer
// Separator behavior that produces a block whose stored shared prefix is
// shorter than srcPrefix.
func (d *DB) validateInSpanBlocks(
	ctx context.Context,
	m *manifest.TableMetadata,
	blocks []blockInfo,
	srcPrefix []byte,
	blobContext sstable.TableBlobContext,
) error {
	if len(blocks) == 0 {
		return nil
	}
	return d.fileCache.withReader(ctx, block.NoReadEnv, m,
		func(r *sstable.Reader, _ sstable.ReadEnv) error {
			for _, b := range blocks {
				first, last, err := r.FirstAndLastUserKeyOfDataBlock(ctx, b.handle.Handle, blobContext)
				if err != nil {
					return errors.Wrapf(err,
						"pebble: VirtualClone failed reading data block of source %s", m.TableNum)
				}
				if !bytes.HasPrefix(first, srcPrefix) || !bytes.HasPrefix(last, srcPrefix) {
					return errors.Wrapf(ErrUnsupportedClone,
						"source table %s has an in-span data block whose key range is not entirely within srcPrefix %q (first=%q last=%q)",
						m.TableNum, srcPrefix, first, last)
				}
			}
			return nil
		})
}

// firstAndLastKeyOfRun returns the first InternalKey of firstBlock and the
// last InternalKey of lastBlock, in storage-prefix space.
func (d *DB) firstAndLastKeyOfRun(
	ctx context.Context,
	m *manifest.TableMetadata,
	firstBlock, lastBlock blockInfo,
	blobContext sstable.TableBlobContext,
) (first, last base.InternalKey, _ error) {
	err := d.fileCache.withReader(ctx, block.NoReadEnv, m,
		func(r *sstable.Reader, _ sstable.ReadEnv) error {
			f, l, err := r.FirstAndLastInternalKeyOfDataBlock(ctx, firstBlock.handle.Handle, blobContext)
			if err != nil {
				return err
			}
			first = f
			if firstBlock.handle.Offset == lastBlock.handle.Offset {
				last = l
				return nil
			}
			_, l, err = r.FirstAndLastInternalKeyOfDataBlock(ctx, lastBlock.handle.Handle, blobContext)
			if err != nil {
				return err
			}
			last = l
			return nil
		})
	return first, last, err
}

// rewriteBoundaryBlock decodes the boundary block referenced by bh, selects
// the keys whose user keys lie within srcSpan, translates each to dst space,
// and writes them to a new physical SST. Returns nil for the metadata if no
// in-span keys exist (degenerate case).
//
// The returned DiskFileNum identifies the object so the caller can clean it
// up if a later step fails before the VE applies.
func (d *DB) rewriteBoundaryBlock(
	ctx context.Context,
	m *manifest.TableMetadata,
	bh block.Handle,
	srcSpan KeyRange,
	srcSpanBounds base.UserKeyBounds,
	srcPrefix, dstPrefix []byte,
	level int,
	blobContext sstable.TableBlobContext,
) (*manifest.TableMetadata, base.DiskFileNum, error) {
	cmp := d.cmp

	// Collect (translatedKey, value) pairs for in-span keys. Values from
	// out-of-line storage (value blocks or blob files) are materialized inline
	// into the new physical SST: blobContext supplies the blob fetcher and
	// the source SST's BlobReferences mapping. The new boundary SST has no
	// blob references of its own; the source blob file's refcount is
	// unaffected (the source SST keeps its reference).
	type kvPair struct {
		key   base.InternalKey
		value []byte
	}
	var pairs []kvPair
	if err := d.fileCache.withReader(ctx, block.NoReadEnv, m,
		func(r *sstable.Reader, _ sstable.ReadEnv) error {
			return r.IterateDataBlock(ctx, bh, blobContext, func(k base.InternalKey, v []byte) error {
				if !srcSpanBounds.ContainsInternalKey(cmp, k) {
					return nil
				}
				translated := translateInternalKey(srcPrefix, dstPrefix, k)
				pairs = append(pairs, kvPair{
					key:   translated,
					value: append([]byte(nil), v...),
				})
				return nil
			})
		}); err != nil {
		return nil, 0, errors.Wrapf(err,
			"pebble: VirtualClone failed iterating boundary block of source %s", m.TableNum)
	}
	if len(pairs) == 0 {
		return nil, 0, nil
	}

	// Allocate a new physical file and write the SST.
	tableNum := d.mu.versions.getNextTableNum()
	fileNum := base.PhysicalTableDiskFileNum(tableNum)

	writable, _, err := d.objProvider.Create(ctx, base.FileTypeTable, fileNum,
		objstorage.CreateOptions{PreferSharedStorage: false})
	if err != nil {
		return nil, 0, errors.Wrapf(err,
			"pebble: VirtualClone failed to create boundary SST object")
	}

	writerOpts := d.opts.MakeWriterOptions(level, d.TableFormat())
	tw := sstable.NewRawWriter(writable, writerOpts)
	for _, p := range pairs {
		if err := tw.Add(p.key, p.value, false /* forceObsolete */, base.KVMeta{}); err != nil {
			_ = tw.Close()
			_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
			return nil, 0, errors.Wrapf(err, "pebble: VirtualClone boundary write")
		}
	}
	if err := tw.Close(); err != nil {
		_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
		return nil, 0, errors.Wrapf(err, "pebble: VirtualClone boundary close")
	}
	wm, err := tw.Metadata()
	if err != nil {
		_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
		return nil, 0, errors.Wrapf(err, "pebble: VirtualClone boundary metadata")
	}

	pm := &manifest.TableMetadata{
		TableNum:              tableNum,
		Size:                  wm.Size,
		SeqNums:               m.SeqNums,
		LargestSeqNumAbsolute: m.LargestSeqNumAbsolute,
	}
	pm.ExtendPointKeyBounds(d.cmp, wm.SmallestPoint, wm.LargestPoint)
	pm.InitPhysicalBacking()
	if err := pm.Validate(d.cmp, d.opts.Comparer.FormatKey); err != nil {
		_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
		return nil, 0, errors.Wrapf(err,
			"pebble: VirtualClone boundary table %s is invalid", pm.TableNum)
	}
	return pm, fileNum, nil
}
