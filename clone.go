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
var ErrUnsupportedClone = errors.New("pebble: unsupported VirtualClone case")

// virtualCloneMaxRetries bounds the number of times VirtualClone will retry
// after losing a race against a concurrent compaction or excise that mutated
// the source backings between the read phase and the apply phase.
const virtualCloneMaxRetries = 5

// VirtualClone walks every SST in the current LSM that intersects srcSpan and
// creates virtual SSTs (sharing existing physical backings) that expose the
// intersected keys under dstPrefix instead of srcPrefix.
//
// Parameter encoding contract:
//
// ┌───────────────────┬──────────────────────────────┬──────────────────────────────────────┐
//
//	│ Parameter         │ Encoding                     │ Why                                  │
//
// ├───────────────────┼──────────────────────────────┼──────────────────────────────────────┤
//
//	│ srcPrefix         │ Raw bytes (no Comparer       │ This is the literal byte prefix      │
//	│ dstPrefix         │   encoding, no sentinel).    │ BlockPrefixSubstitution will strip   │
//	│                   │ Equal-length required.       │ from / replace at the start of every │
//	│                   │                              │ in-block stored key. It must equal   │
//	│                   │                              │ exactly the bytes physically present │
//	│                   │                              │ at the start of stored keys.         │
//
// ├───────────────────┼──────────────────────────────┼──────────────────────────────────────┤
//
//	│ srcSpan.Start/End │ Comparer-encoded keys (e.g.  │ These are *keys* used in compare     │
//	│ dstSpan.Start/End │   for CRDB:                  │ operations against SST bounds and    │
//	│                   │   EngineKey{...}.Encode()).  │ memtable contents; they must be      │
//	│                   │                              │ well-formed under the active         │
//	│                   │                              │ Comparer (which may interpret e.g.   │
//	│                   │                              │ a trailing byte as a suffix length). │
//
// └───────────────────┴──────────────────────────────┴──────────────────────────────────────┘
//
// Validation enforced by validateVirtualCloneInputs:
//
//   - srcPrefix and dstPrefix are non-empty, distinct, equal-length.
//   - bytes.HasPrefix(srcSpan.Start, srcPrefix). (srcSpan.End is intentionally
//     not constrained — the canonical CRDB shape [prefix, prefix.PrefixEnd())
//     puts End outside the prefix.)
//   - bytes.HasPrefix(dstSpan.Start, dstPrefix).
//   - dstSpan.Start == byte-translate(srcSpan.Start, srcPrefix → dstPrefix).
//     This locks the dst-side lower edge to the substitution image of the
//     src-side lower edge so cloned bounds (which lie inside dstSpan by
//     construction) never escape dstSpan on the low side.
//
// dstSpan is the destination keyspan that VirtualClone owns: the post-
// condition is that dstSpan is a snapshot of srcSpan in dst space.
// VirtualClone atomically excises dstSpan against the live LSM (regardless of
// what, if anything, is cloned into it) and installs the cloned virtual /
// physical SSTs in the same VersionEdit. dstSpan.End is caller-provided
// because Pebble cannot synthesize a Comparer-safe upper bound from a
// prefix-swap alone (e.g., bytesPrefixEnd of `\xfe\x8c` produces `\xfe\x8d`,
// which CRDB's cockroachkvs Comparer reads as a 0x8d-byte MVCC suffix on a
// 2-byte key and panics).
//
// Atomicity: a single VersionEdit installs all virtual SSTs and any small
// physical SSTs produced for boundary-block rewrites. On conflict with a
// concurrent compaction/excise on a referenced source SST, the operation
// restarts (bounded retry); orphaned boundary SSTs from the failed attempt
// are cleaned up.
//
// Level placement: cloned files are placed at their source levels. Because
// dstSpan is excised in the same VE, the dst region is empty at every level
// before the cloned files install, so source-level placement cannot conflict
// with any surviving file.
//
// Returns ErrUnsupportedClone (with details) for v1 unsupported cases:
//   - rowblk-format SSTs intersecting srcSpan
//   - a source SST that contains a range deletion or range key whose
//     [Start, End) interval straddles a srcSpan boundary
//   - a straddling source SST whose in-span data blocks have a stored
//     shared prefix shorter than srcPrefix
func (d *DB) VirtualClone(
	ctx context.Context, srcSpan KeyRange, srcPrefix []byte, dstSpan KeyRange, dstPrefix []byte,
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
	if err := validateVirtualCloneInputs(d.opts.Comparer, srcSpan, srcPrefix, dstSpan, dstPrefix); err != nil {
		return err
	}

	// Each attempt has two phases:
	//
	//   Phase A (no commit-pipeline sem held): snapshot the LSM, walk source
	//     SSTs, build the cloned plan entries (writing any boundary-block
	//     physical SSTs to disk).
	//   Phase B (under commit.AllocateSeqNum): allocate the excise seqnum,
	//     flush memtables overlapping srcSpan/dstSpan/EFOS-protected ranges,
	//     and install the version edit (cloned entries + dst-span excise).
	//
	// Splitting the phases keeps the read+build work outside the commit
	// pipeline's serialization so test hooks (and, in principle, anything
	// else) can issue regular DB writes without deadlocking against the
	// in-flight allocation.
	//
	// On a phase-B abort (a source SST observed in phase A is no longer at
	// its source level when phase B re-validates), we cleanup any boundary
	// SSTs phase A wrote and retry. Each retry consumes a fresh excise
	// seqnum; the abandoned ones leave a small gap in the seqnum sequence,
	// which is harmless.
	for attempt := 0; attempt < virtualCloneMaxRetries; attempt++ {
		entries, preVEObjects, snapshotVersion, buildErr := d.buildClonePlan(ctx, attempt,
			srcSpan, srcPrefix, dstSpan, dstPrefix)
		if buildErr != nil {
			d.cleanupClonePreVEObjects(preVEObjects)
			return buildErr
		}

		aborted, installErr := d.installClonePlanViaCommitPipeline(
			ctx, attempt, entries, srcSpan, dstSpan, snapshotVersion)
		if installErr != nil {
			d.cleanupClonePreVEObjects(preVEObjects)
			return installErr
		}
		if !aborted {
			return nil
		}
		d.cleanupClonePreVEObjects(preVEObjects)
	}
	return errors.Newf(
		"pebble: VirtualClone exhausted %d retries due to concurrent LSM mutations",
		virtualCloneMaxRetries)
}

// installClonePlanViaCommitPipeline runs phase B of one VirtualClone attempt:
// it allocates the excise seqnum through the commit pipeline, force-flushes
// memtables that overlap srcSpan/dstSpan/EFOS-protected ranges, and (under
// the apply callback) installs the cloned entries plus the dstSpan excise.
//
// The allocated seqnum is registered in ongoingExcises so any EFOS created
// while phase B is in flight observes a visibleSeqNum past the excise
// (preserving its pre-excise view); the registration is cleared after
// AllocateSeqNum returns. Mirrors the ingest+excise pattern in DB.ingest.
func (d *DB) installClonePlanViaCommitPipeline(
	ctx context.Context,
	attempt int,
	entries []clonePlanEntry,
	srcSpan, dstSpan KeyRange,
	snapshotVersion *manifest.Version,
) (aborted bool, _ error) {
	var (
		prepareErr     error
		installErr     error
		assignedSeqNum base.SeqNum
		mem            *flushableEntry
		mut            *memTable
	)
	prepare := func(seqNum base.SeqNum) {
		assignedSeqNum = seqNum
		d.mu.Lock()
		defer d.mu.Unlock()
		if _, ok := d.mu.snapshots.ongoingExcises[seqNum]; ok {
			panic(errors.AssertionFailedf("pebble: VirtualClone excise with seqnum %s already in map", seqNum))
		}
		d.mu.snapshots.ongoingExcises[seqNum] = dstSpan

		overlapBounds := []bounded{&srcSpan, &dstSpan}
		overlapBounds = append(overlapBounds,
			exciseOverlapBounds(d.cmp, &d.mu.snapshots.snapshotList, dstSpan, seqNum)...)

		for i := len(d.mu.mem.queue) - 1; i >= 0; i-- {
			m := d.mu.mem.queue[i]
			var anyOverlaps bool
			m.computePossibleOverlaps(func(b bounded) shouldContinue {
				anyOverlaps = true
				return stopIteration
			}, overlapBounds...)
			if !anyOverlaps {
				continue
			}
			if mem == nil {
				mem = m
			}
			if m.flushable == d.mu.mem.mutable {
				if err := d.makeRoomForWrite(nil); err != nil {
					prepareErr = err
					return
				}
			}
			m.flushForced = true
		}
		if mem == nil {
			// No overlap; ref the mutable memtable as a writer to prevent
			// it (and any later memtables) from flushing before the apply
			// installs the excise. Mirrors ingest.go's mut handling.
			mut = d.mu.mem.mutable
			mut.writerRef()
		} else {
			d.maybeScheduleFlush()
		}
	}
	apply := func(seqNum base.SeqNum) {
		defer func() {
			if mut != nil && mut.writerUnref() {
				d.mu.Lock()
				d.maybeScheduleFlush()
				d.mu.Unlock()
			}
		}()
		if prepareErr != nil {
			installErr = prepareErr
			return
		}
		if mem != nil {
			// The phase-A build ran on a snapshot taken before this flush
			// completed, so the freshly-flushed memtable contents are not
			// reflected in `entries`. Wait for the flush to land, then
			// abort: the next iteration's phase A will snapshot the post-
			// flush LSM and pick up the flushed SSTs. Steady-state cost is
			// at most one extra retry per VirtualClone call.
			select {
			case <-mem.flushed:
			case <-ctx.Done():
				installErr = ctx.Err()
				return
			}
			aborted = true
			return
		}
		// Check if the LSM gained new files overlapping srcSpan since
		// phase A's snapshot. A background flush between phase A and B
		// could move data from memtable to L0 without triggering the
		// abort-on-flush path (the memtable is already gone from the
		// queue by the time prepare runs). If new srcSpan files appeared,
		// abort and retry so phase A picks them up.
		d.mu.Lock()
		if curV := d.mu.versions.currentVersion(); curV != snapshotVersion {
			srcBounds := srcSpan.UserKeyBounds()
			for layer, ls := range curV.AllLevelsAndSublevels() {
				for m := range ls.Overlaps(d.cmp, srcBounds).All() {
					if !snapshotVersion.Contains(layer.Level(), m) {
						d.mu.Unlock()
						aborted = true
						return
					}
				}
			}
		}
		d.mu.Unlock()
		a, err := d.installClonePlan(ctx, attempt, entries, srcSpan, dstSpan, seqNum)
		aborted = a
		installErr = err
	}
	d.commit.AllocateSeqNum(1, prepare, apply)
	// NB: removeFromOngoingExcises must happen after AllocateSeqNum returns
	// (i.e. after the assigned seqnum has been published as visible), so
	// that any concurrent EFOS creation that grabs DB.mu after the removal
	// observes a visibleSeqNum past our excise. Mirrors ingest.go:2042-2048.
	d.removeFromOngoingExcises(assignedSeqNum)
	return aborted, installErr
}

// cleanupClonePreVEObjects removes physical SSTs that a clone attempt wrote
// before its version edit applied. Used on abort/retry and on terminal errors
// to avoid leaving orphaned boundary-block files in the object provider.
func (d *DB) cleanupClonePreVEObjects(preVEObjects []base.DiskFileNum) {
	for _, fn := range preVEObjects {
		_ = d.objProvider.Remove(base.FileTypeTable, fn)
	}
}

// validateVirtualCloneInputs checks the static preconditions of VirtualClone.
func validateVirtualCloneInputs(
	cmp *base.Comparer, srcSpan KeyRange, srcPrefix []byte, dstSpan KeyRange, dstPrefix []byte,
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
	if !dstSpan.Valid() {
		return errors.New("pebble: VirtualClone requires a valid dstSpan")
	}
	if cmp.Compare(dstSpan.Start, dstSpan.End) >= 0 {
		return errors.Newf("pebble: VirtualClone dstSpan start %q is not before end %q",
			dstSpan.Start, dstSpan.End)
	}
	if !bytes.HasPrefix(dstSpan.Start, dstPrefix) {
		return errors.Newf(
			"pebble: VirtualClone dstSpan start %q does not have dstPrefix %q",
			dstSpan.Start, dstPrefix)
	}
	// dstSpan.Start must be the bytewise translation of srcSpan.Start under
	// the srcPrefix→dstPrefix substitution. This is the same translation that
	// the cloned virtual SSTs apply to source keys, so making the excised
	// region's lower edge match keeps the cloned bounds inside dstSpan.
	expectedDstStart := make([]byte, 0, len(dstPrefix)+len(srcSpan.Start)-len(srcPrefix))
	expectedDstStart = append(expectedDstStart, dstPrefix...)
	expectedDstStart = append(expectedDstStart, srcSpan.Start[len(srcPrefix):]...)
	if !bytes.Equal(dstSpan.Start, expectedDstStart) {
		return errors.Newf(
			"pebble: VirtualClone dstSpan start %q does not match srcSpan start %q translated under srcPrefix→dstPrefix (expected %q)",
			dstSpan.Start, srcSpan.Start, expectedDstStart)
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

// translateBoundaryUserKey is like translateUserKey but additionally maps
// the special case key == srcSpan.End to dstSpan.End. This is needed for
// fragment bounds that have been truncated to srcSpan: a rangedel/rangekey
// fragment originally spanning [A, B) where B > srcSpan.End is truncated
// to [max(A, srcSpan.Start), srcSpan.End). The truncated end key srcSpan.End
// generally does NOT have srcPrefix as a bytes-prefix (e.g. when the caller
// asks to clone all of srcPrefix and passes srcSpan.End as the upper bound
// of srcPrefix's keyspace), so a bytewise srcPrefix→dstPrefix substitution
// would be ill-defined. The dst-space analog of srcSpan.End is dstSpan.End,
// which the caller has supplied via the API.
func translateBoundaryUserKey(
	srcPrefix, dstPrefix []byte, srcSpan, dstSpan KeyRange, key []byte,
) []byte {
	if bytes.Equal(key, srcSpan.End) {
		return append([]byte(nil), dstSpan.End...)
	}
	return translateUserKey(srcPrefix, dstPrefix, key)
}

// translateBoundaryInternalKey is the InternalKey form of translateBoundaryUserKey.
func translateBoundaryInternalKey(
	srcPrefix, dstPrefix []byte, srcSpan, dstSpan KeyRange, k base.InternalKey,
) base.InternalKey {
	return base.InternalKey{
		UserKey: translateBoundaryUserKey(srcPrefix, dstPrefix, srcSpan, dstSpan, k.UserKey),
		Trailer: k.Trailer,
	}
}

// clonePlanEntry describes one cloned file (virtual or physical) to install.
type clonePlanEntry struct {
	// sourceLevel is the level of the originating source SST (the floor for
	// destination-level placement).
	sourceLevel int
	// source is the source TableMetadata. Always set for entries derived from
	// a single source SST (data-run virtual, boundary-rewrite physical,
	// fragment-rewrite physical). Used for source-level/backing bookkeeping
	// and for grouping entries when forceL0 placement of one entry must
	// propagate to its siblings.
	source *manifest.TableMetadata
	// virtual is the virtual TableMetadata to install. Mutually exclusive with
	// physical.
	virtual *manifest.TableMetadata
	// physical is the physical TableMetadata to install (boundary-block
	// rewrite or fragment-rewrite). Mutually exclusive with virtual. Physical
	// entries are self-contained: they have their own backing initialized by
	// InitPhysicalBacking and do not share the source's backing.
	physical *manifest.TableMetadata
	// assignedLevel is the level chosen by placeClonedFiles. -1 if not yet
	// assigned.
	assignedLevel int
	// forceL0 places this entry at L0 regardless of its source's level. Used
	// for fragment-rewrite physical SSTs whose bounds (the truncated extent
	// of a wide range deletion or range key) may overlap the cloned virtual
	// SST's data-run bounds and the boundary-rewrite physical SSTs at the
	// source level. L0 admits overlapping files via sublevels.
	forceL0 bool
}

// buildClonePlan handles phase A of one VirtualClone attempt: snapshot the
// LSM, walk source SSTs intersecting srcSpan, validate fragments, and build
// the cloned plan entries (including writing any boundary-block physical
// SSTs to disk). Runs without holding the commit pipeline semaphore so test
// hooks (and concurrent operations more generally) may freely commit.
//
// On any error, returned preVEObjects identifies any boundary SSTs already
// written that the caller must remove from the object provider.
func (d *DB) buildClonePlan(
	ctx context.Context,
	attempt int,
	srcSpan KeyRange,
	srcPrefix []byte,
	dstSpan KeyRange,
	dstPrefix []byte,
) (
	entries []clonePlanEntry,
	preVEObjects []base.DiskFileNum,
	snapshotVersion *manifest.Version,
	_ error,
) {
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

	// Shared blob value fetcher for the lifetime of this build. Used by
	// boundary-block reads (and any other block-level helper) to materialize
	// values whose handles point into blob files. Per-source TableBlobContexts
	// constructed downstream wire this fetcher together with the source
	// table's own BlobReferences.
	var blobFetcher blob.ValueFetcher
	blobFetcher.Init(&currentVersion.BlobFiles, d.fileCache,
		block.ReadEnv{}, blob.SuggestedCachedReaders(currentVersion.MaxReadAmp()))
	defer func() { _ = blobFetcher.Close() }()

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
				return nil, preVEObjects, nil, err
			}
			// Fully-contained source SSTs go through buildFullyContainedVirtual,
			// which uses BlockPrefixSubstitution to expose every key under
			// dstPrefix. The substitution requires every stored user key to
			// have srcPrefix as its leading bytes — including the SST's
			// metadata bounds (smallest/largest), which `buildFullyContainedVirtual`
			// translates verbatim. A subtle case: an SST whose Largest is an
			// exclusive sentinel exactly at srcSpan.End (e.g., a range-del
			// fragment ending at /Tenant/4 when srcSpan.End = /Tenant/4)
			// satisfies `srcSpanBounds.ContainsInternalKey` (the exclusive-
			// sentinel rule treats key.UserKey == End.Key as inside when the
			// trailer is an exclusive sentinel) but its Largest.UserKey does
			// NOT have srcPrefix. Treating such an SST as fully-contained and
			// substituting verbatim would corrupt the bounds. Routing it to
			// the straddler path lets `rewriteStraddlerFragments` clip the
			// rangedel to srcSpan and translate via the boundary mapping
			// (srcSpan.End → dstSpan.End), which is the correct shape.
			fullyContained := srcSpanBounds.ContainsInternalKey(d.cmp, m.Smallest()) &&
				srcSpanBounds.ContainsInternalKey(d.cmp, m.Largest()) &&
				bytes.HasPrefix(m.Smallest().UserKey, srcPrefix) &&
				bytes.HasPrefix(m.Largest().UserKey, srcPrefix)
			if fullyContained {
				vm, err := d.buildFullyContainedVirtual(m, srcPrefix, dstPrefix)
				if err != nil {
					return nil, preVEObjects, nil, err
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
				ctx, m, level, srcSpan, srcSpanBounds, srcPrefix, dstSpan, dstPrefix, survey, &blobFetcher)
			preVEObjects = append(preVEObjects, written...)
			if err != nil {
				return nil, preVEObjects, nil, err
			}
			entries = append(entries, straddlerEntries...)
		}
	}

	// A source SST whose bounds extend into dstSpan (a single physical file
	// covering both src-prefix and dst-prefix keys — common after a snapshot
	// receive into a region neighboring an existing tenant) is handled by
	// deferring its stand-in / DeletedTables / CreatedBackingTables to the
	// phase-B dst-excise loop. exciseTable + applyExciseToVersionEdit will
	// trim the file at dstSpan.Start, producing a virtual leftTable that
	// covers the surviving src-side region — which IS the stand-in for
	// those files. The cloned-virtual / boundary-rewrite entries this phase
	// builds for the in-srcSpan data are unaffected (they live in dst space
	// and don't overlap the leftTable). See installClonePlan for the skip
	// logic that avoids the otherwise-conflicting double install.

	// Each cloned entry is placed at its source level (after dst-excise wipes
	// dstSpan, no surviving file can conflict). Fragment-rewrite physical
	// SSTs (forceL0) go to L0 instead: their bounds — the truncated extent
	// of a wide range deletion or range key — typically encompass the data-
	// run virtual SST and any boundary-rewrite physical SSTs from the same
	// source, and L0's sublevel structure tolerates that overlap whereas
	// L1+ does not.
	//
	// When a source produces a forceL0 fragment-rewrite SST, ALL other
	// entries from that same source (the in-span data run virtual, any
	// boundary-rewrite physicals) are also forced to L0. Reason: the cloned
	// data SST gets its trailers bumped to exciseSeqNum (via SyntheticSeqNum)
	// in installClonePlan, but the fragment-rewrite SST keeps its source
	// rangedel seqnums (which may be lower). Splitting the cloned data to a
	// lower level (e.g., L6) than the rangedel (L0) violates the LSM
	// invariant that higher levels carry newer seqnums; the merging iter's
	// seek-past-tombstone optimization (merging_iter.go:639) doesn't check
	// seqnums for higher-level tombstones — it assumes the LSM invariant —
	// and would seek the cloned data iterator past the tombstone end,
	// silently eliding the cloned data. Co-locating all entries from a
	// single source in L0 lets sublevel placement (sorted by LargestSeqNum)
	// put the cloned data SST in a NEWER sublevel (lower mergingIter index)
	// than the rangedel SST, so the seek-past loop never visits the rangedel
	// when the data item is heap top.
	sourcesWithForceL0 := make(map[base.TableNum]struct{})
	for i := range entries {
		if entries[i].forceL0 && entries[i].source != nil {
			sourcesWithForceL0[entries[i].source.TableNum] = struct{}{}
		}
	}
	for i := range entries {
		if entries[i].forceL0 {
			entries[i].assignedLevel = 0
			continue
		}
		if entries[i].source != nil {
			if _, ok := sourcesWithForceL0[entries[i].source.TableNum]; ok {
				entries[i].assignedLevel = 0
				continue
			}
		}
		entries[i].assignedLevel = entries[i].sourceLevel
	}
	return entries, preVEObjects, currentVersion, nil
}

// installClonePlan handles phase B of one VirtualClone attempt: re-validates
// the plan against the live version (aborts if any source SST has moved out
// from under the snapshot) and installs the version edit containing all
// cloned entries plus the dstSpan excise. Called from inside the apply
// callback of commit.AllocateSeqNum so the excise carries a properly-ordered
// seqnum.
func (d *DB) installClonePlan(
	ctx context.Context,
	attempt int,
	entries []clonePlanEntry,
	srcSpan KeyRange,
	dstSpan KeyRange,
	exciseSeqNum base.SeqNum,
) (aborted bool, _ error) {
	dstSpanBounds := dstSpan.UserKeyBounds()
	srcSpanBounds := srcSpan.UserKeyBounds()
	_ = srcSpanBounds

	// Test hook: invoked just before re-acquiring d.mu and applying the
	// version edit.
	if hook := d.opts.private.testingCloneBeforeUpdateVersionLocked; hook != nil {
		hook(attempt)
	}

	d.mu.Lock()
	jobID := d.newJobIDLocked()
	defer d.mu.Unlock()

	var appliedVE *manifest.VersionEdit
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
			// Bump the cloned point-data SST's effective seqnum to the excise
			// seqnum (allocated by AllocateSeqNum, post-dates all source data).
			// Without this, source-backing data whose key trailers were
			// snap-zeroed at compaction time (e.g. for an L6 SST below the
			// earliest snapshot) ends up in dst at trailer seqnum 0, where any
			// rangedel at the same seqnum (kind ordering: DEL > SET) shadows
			// it on read. We bump only the cloned point-data SSTs (cloned
			// virtual + boundary-rewrite physicals), NOT the fragment-rewrite
			// physical (e.forceL0): the fragment rewrite preserves source
			// rangedel seqnums so cloned data sits ABOVE cloned tombstones.
			// SyntheticSeqNum activates when SeqNums.Low == SeqNums.High.
			if !e.forceL0 {
				meta.SeqNums = base.SeqNumRange{Low: exciseSeqNum, High: exciseSeqNum}
				if meta.LargestSeqNumAbsolute < exciseSeqNum {
					meta.LargestSeqNumAbsolute = exciseSeqNum
				}
				// SyntheticSeqNum rewrites all key trailers to exciseSeqNum.
				// Update PointKeyBounds and RangeKeyBounds trailers to match,
				// so assertIter (table-stats) and other bounds checks remain
				// consistent. Skip exclusive sentinels (SeqNumMax).
				if meta.HasPointKeys {
					s := meta.PointKeyBounds.Smallest()
					l := meta.PointKeyBounds.Largest()
					s.Trailer = base.MakeTrailer(exciseSeqNum, s.Trailer.Kind())
					if !l.IsExclusiveSentinel() {
						l.Trailer = base.MakeTrailer(exciseSeqNum, l.Trailer.Kind())
					}
					meta.PointKeyBounds.SetInternalKeyBounds(s, l)
				}
				if meta.HasRangeKeys {
					s := meta.RangeKeyBounds.Smallest()
					l := meta.RangeKeyBounds.Largest()
					s.Trailer = base.MakeTrailer(exciseSeqNum, s.Trailer.Kind())
					if !l.IsExclusiveSentinel() {
						l.Trailer = base.MakeTrailer(exciseSeqNum, l.Trailer.Kind())
					}
					meta.RangeKeyBounds.SetInternalKeyBounds(s, l)
				}
				meta.RecomputeOverallBoundTypes(d.cmp)
			}
			ve.NewTables = append(ve.NewTables, manifest.NewTableEntry{
				Level: e.assignedLevel,
				Meta:  meta,
			})
			if e.physical != nil {
				// Boundary-rewrite and fragment-rewrite physical SSTs are
				// self-contained: they have a fresh backing initialized by
				// InitPhysicalBacking and no source-backing sharing to
				// arrange. (e.source is still set on these entries — used
				// only for forceL0 sibling-grouping in placeClonedFiles.)
				continue
			}
			if e.source == nil {
				continue
			}
			if _, ok := seenSource[e.source.TableNum]; ok {
				continue
			}
			seenSource[e.source.TableNum] = struct{}{}

			// If this source SST's bounds extend into dstSpan, the dst-excise
			// loop below will run exciseTable on it and produce a virtual
			// leftTable over the same backing — which IS the stand-in for
			// the surviving src-side region — and applyExciseToVersionEdit
			// will add the file to DeletedTables and (if physical) append
			// its backing to CreatedBackingTables. Skip the equivalent work
			// here so we don't install a duplicate stand-in or double-add
			// CreatedBackingTables (the manifest invariant requires at most
			// one CreatedBackingTables entry per backing). The cloned
			// virtual SSTs / boundary rewrites we already appended above
			// are unaffected: they live in dst space and don't overlap
			// the leftTable.
			if dstSpanBounds.Overlaps(d.cmp, e.source.UserKeyBounds()) {
				continue
			}

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
		}
		// Cancel any in-progress compaction whose bounds overlap dstSpan
		// or srcSpan. dstSpan: a compaction may write a file landing in
		// dstSpan after the excise applies, defeating it. srcSpan: the
		// stand-in replacement path replaces source physical SSTs with
		// virtual stand-ins; a compaction whose inputs include that source
		// would commit with a DeletedTables entry for a file no longer in
		// the level.
		if anyExcised || len(seenSource) > 0 {
			for c := range d.mu.compact.inProgress {
				if c.VersionEditApplied() {
					continue
				}
				bounds := c.Bounds()
				if bounds != nil && (bounds.Overlaps(d.cmp, dstSpanBounds) ||
					bounds.Overlaps(d.cmp, srcSpanBounds)) {
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

		appliedVE = ve
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
	d.fireCloneEvents(int(jobID), appliedVE)
	d.updateReadStateLocked(d.opts.DebugCheck)
	if appliedVE != nil {
		d.updateTableStatsLocked(appliedVE.NewTables)
	}
	return false, nil
}

// fireCloneEvents emits EventListener notifications for the tables and
// backings that VirtualClone added/removed via ve. VirtualClone bypasses the
// flush/compaction/ingest paths that normally fire these events, so without
// these calls cloned files are invisible to consumers of the storage event
// stream (CockroachDB's storage log, the metamorphic harness, etc).
func (d *DB) fireCloneEvents(jobID int, ve *manifest.VersionEdit) {
	if ve == nil {
		return
	}
	if d.opts.EventListener.TableCreated != nil {
		for i := range ve.NewTables {
			meta := ve.NewTables[i].Meta
			fileNum := meta.TableBacking.DiskFileNum
			objMeta, err := d.objProvider.Lookup(base.FileTypeTable, fileNum)
			path := ""
			if err == nil {
				path = d.objProvider.Path(objMeta)
			}
			d.opts.EventListener.TableCreated(TableCreateInfo{
				JobID:   jobID,
				Reason:  "virtual-clone",
				Path:    path,
				FileNum: fileNum,
			})
		}
	}
	if d.opts.EventListener.TableDeleted != nil {
		for _, m := range ve.DeletedTables {
			d.opts.EventListener.TableDeleted(TableDeleteInfo{
				JobID:   jobID,
				FileNum: m.TableBacking.DiskFileNum,
			})
		}
	}
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
			// NB: SuppressUnderlyingKeyspans is intentionally false here.
			// `buildFullyContainedVirtual` is the sole source of truth for
			// the cloned dst-space view of this source — there is no L0
			// fragment SST sibling, so the standIn must surface the
			// underlying physical's keyspan blocks (translated via
			// substitution) to expose any in-srcPrefix range-del / range-key
			// fragments.
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
	srcPrefix []byte,
	dstSpan KeyRange,
	dstPrefix []byte,
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
		// Pebble's flush may drop SET keys obsoleted by a same-batch
		// RANGEDEL with a higher seqnum, producing an SST whose only
		// content is the rangedel block (no data blocks). Such a source
		// has no in-span point keys to clone; the only thing to materialize
		// is the in-srcSpan portion of its fragments. Fall through to the
		// fragment-rewrite path below.
		if survey.hasInSpanRangeDel || survey.hasInSpanRangeKey {
			fragMeta, fragFileNum, fragErr := d.rewriteStraddlerFragments(
				ctx, m, srcSpan, srcPrefix, dstSpan, dstPrefix, level)
			if fragErr != nil {
				return entries, written, fragErr
			}
			if fragMeta != nil {
				written = append(written, fragFileNum)
				entries = append(entries, clonePlanEntry{
					sourceLevel:   level,
					source:        m,
					physical:      fragMeta,
					assignedLevel: -1,
					forceL0:       true,
				})
			}
		}
		return entries, written, nil
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
				Src:                        append([]byte(nil), srcPrefix...),
				Dst:                        append([]byte(nil), dstPrefix...),
				SuppressUnderlyingKeyspans: true,
			},
		}
		vm.ExtendPointKeyBounds(d.cmp, smallest, largest)
		// NB: do NOT extend the virtual SST's bounds to encompass in-span
		// range-deletion / range-key fragments. A wide rangedel can produce
		// bounds that overlap the boundary-rewrite physical SSTs from this
		// same source (and the cloned virtual SST + boundary SSTs all live
		// at the source level, where pebble forbids overlapping files).
		// In-span fragments are materialized into a separate physical SST by
		// rewriteStraddlerFragments and placed at L0; see below.

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
	}
	// In-span fragments (range-del / range-key) are materialized into a
	// separate physical SST and placed at L0. See rewriteStraddlerFragments
	// for the rationale.
	if survey.hasInSpanRangeDel || survey.hasInSpanRangeKey {
		fragMeta, fragFileNum, err := d.rewriteStraddlerFragments(
			ctx, m, srcSpan, srcPrefix, dstSpan, dstPrefix, level)
		if err != nil {
			return entries, written, err
		}
		if fragMeta != nil {
			written = append(written, fragFileNum)
			entries = append(entries, clonePlanEntry{
				sourceLevel:   level,
				source:        m,
				physical:      fragMeta,
				assignedLevel: -1,
				forceL0:       true,
			})
		}
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
			source:        m,
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
	// walk iterates the fragments in iter, classifies each relative to srcSpan,
	// and updates the survey for any fragment that has any overlap with
	// srcSpan. The survey records the bounds of the in-span (clipped-to-
	// srcSpan) fragment extents in src space — used by callers to decide
	// whether to materialize a fragment-rewrite physical SST and to
	// short-circuit when no in-span fragments exist. Straddlers are not
	// rejected; the caller is responsible for materializing the truncated
	// (in-srcSpan) portion via rewriteStraddlerFragments.
	walk := func(
		iter keyspan.FragmentIterator,
		updateBounds func(effSmallest, effLargest base.InternalKey, kind base.InternalKeyKind, trailer base.InternalKeyTrailer),
	) error {
		if iter == nil {
			return nil
		}
		defer iter.Close()
		for s, err := iter.First(); s != nil || err != nil; s, err = iter.Next() {
			if err != nil {
				return errors.Wrapf(err,
					"pebble: VirtualClone failed iterating fragments of source table %s",
					m.TableNum)
			}
			// Skip fully-outside fragments.
			if cmp(s.End, srcSpan.Start) <= 0 || cmp(s.Start, srcSpan.End) >= 0 {
				continue
			}
			effStart := s.Start
			if cmp(effStart, srcSpan.Start) < 0 {
				effStart = srcSpan.Start
			}
			effEnd := s.End
			if cmp(effEnd, srcSpan.End) > 0 {
				effEnd = srcSpan.End
			}
			smallest := base.InternalKey{
				UserKey: append([]byte(nil), effStart...),
				Trailer: s.Keys[0].Trailer,
			}
			updateBounds(smallest,
				base.MakeExclusiveSentinelKey(s.Keys[len(s.Keys)-1].Kind(), append([]byte(nil), effEnd...)),
				s.Keys[0].Kind(), s.Keys[0].Trailer)
		}
		return nil
	}
	_ = level // formerly used in the now-removed straddler error message

	err := d.fileCache.withReader(ctx, block.NoReadEnv, m,
		func(r *sstable.Reader, _ sstable.ReadEnv) error {
			rdel, err := r.NewRawRangeDelIter(ctx, sstable.NoFragmentTransforms, sstable.NoReadEnv)
			if err != nil {
				return errors.Wrapf(err,
					"pebble: VirtualClone failed opening range-del iterator on source %s",
					m.TableNum)
			}
			if err := walk(rdel, func(smallest, largest base.InternalKey, _ base.InternalKeyKind, _ base.InternalKeyTrailer) {
				// Force the largest's kind to RangeDelete (uniform regardless
				// of the source span's keys[len-1] kind, since point-key
				// bounds for range deletions are always RangeDelete-kinded
				// exclusive sentinels).
				largest = base.MakeExclusiveSentinelKey(base.InternalKeyKindRangeDelete, largest.UserKey)
				if !survey.hasInSpanRangeDel {
					survey.hasInSpanRangeDel = true
					survey.smallestRangeDel = smallest
					survey.largestRangeDel = largest
					return
				}
				if base.InternalCompare(cmp, smallest, survey.smallestRangeDel) < 0 {
					survey.smallestRangeDel = smallest
				}
				if base.InternalCompare(cmp, largest, survey.largestRangeDel) > 0 {
					survey.largestRangeDel = largest
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
			return walk(rkey, func(smallest, largest base.InternalKey, _ base.InternalKeyKind, _ base.InternalKeyTrailer) {
				largest = base.MakeExclusiveSentinelKey(base.InternalKeyKindRangeKeyMax, largest.UserKey)
				if !survey.hasInSpanRangeKey {
					survey.hasInSpanRangeKey = true
					survey.smallestRangeKey = smallest
					survey.largestRangeKey = largest
				} else {
					if base.InternalCompare(cmp, smallest, survey.smallestRangeKey) < 0 {
						survey.smallestRangeKey = smallest
					}
					if base.InternalCompare(cmp, largest, survey.largestRangeKey) > 0 {
						survey.largestRangeKey = largest
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

// rewriteStraddlerFragments reads the source SST's range deletion and range
// key fragments, truncates each to srcSpan, translates start/end into dst
// space (using srcSpan.End → dstSpan.End for the boundary case), and writes
// them to a new physical SST. The returned SST contains only fragments — no
// point keys — and replaces the otherwise-needed bounds-extension of the
// cloned virtual SST. Returns nil metadata when no in-span fragments exist.
//
// The new physical SST is intended to be placed at L0 by the caller (via
// clonePlanEntry.forceL0) because its bounds — the truncated extent of the
// fragments — typically encompass the cloned virtual SST and any boundary-
// rewrite physical SSTs from the same source. L0's sublevel structure
// tolerates that overlap whereas L1+ does not.
func (d *DB) rewriteStraddlerFragments(
	ctx context.Context,
	m *manifest.TableMetadata,
	srcSpan KeyRange,
	srcPrefix []byte,
	dstSpan KeyRange,
	dstPrefix []byte,
	level int,
) (*manifest.TableMetadata, base.DiskFileNum, error) {
	cmp := d.cmp

	// Helper: clip a fragment to srcSpan and translate the user-key endpoints
	// to dst space. Endpoints inside srcPrefix translate bytewise; an
	// endpoint exactly equal to srcSpan.End maps to dstSpan.End.
	//
	// Each key's Suffix and Value are deep-copied because the underlying
	// fragment iterator reuses (and, under invariants, mangles) the backing
	// buffers across iterator advances. We collect spans up-front and only
	// write them to the new SST after closing the reader, so retained slices
	// must own their bytes.
	clipAndTranslate := func(s *keyspan.Span) (translated keyspan.Span, ok bool) {
		if cmp(s.End, srcSpan.Start) <= 0 || cmp(s.Start, srcSpan.End) >= 0 {
			return keyspan.Span{}, false
		}
		effStart := s.Start
		if cmp(effStart, srcSpan.Start) < 0 {
			effStart = srcSpan.Start
		}
		effEnd := s.End
		if cmp(effEnd, srcSpan.End) > 0 {
			effEnd = srcSpan.End
		}
		clonedKeys := make([]keyspan.Key, len(s.Keys))
		for i := range s.Keys {
			clonedKeys[i] = keyspan.Key{
				Trailer: s.Keys[i].Trailer,
				Suffix:  append([]byte(nil), s.Keys[i].Suffix...),
				Value:   append([]byte(nil), s.Keys[i].Value...),
			}
		}
		translated = keyspan.Span{
			Start:     translateBoundaryUserKey(srcPrefix, dstPrefix, srcSpan, dstSpan, effStart),
			End:       translateBoundaryUserKey(srcPrefix, dstPrefix, srcSpan, dstSpan, effEnd),
			Keys:      clonedKeys,
			KeysOrder: s.KeysOrder,
		}
		return translated, true
	}

	// Two-phase: collect spans first, then write. Collecting first lets us
	// compute the writer's level / bounds before opening the writable, and
	// avoids holding a reader open across the writer's I/O.
	var rangedelSpans, rangekeySpans []keyspan.Span
	if err := d.fileCache.withReader(ctx, block.NoReadEnv, m,
		func(r *sstable.Reader, _ sstable.ReadEnv) error {
			rdel, err := r.NewRawRangeDelIter(ctx, sstable.NoFragmentTransforms, sstable.NoReadEnv)
			if err != nil {
				return errors.Wrapf(err,
					"pebble: VirtualClone failed opening range-del iterator on source %s",
					m.TableNum)
			}
			if rdel != nil {
				defer rdel.Close()
				for s, err := rdel.First(); s != nil || err != nil; s, err = rdel.Next() {
					if err != nil {
						return errors.Wrapf(err,
							"pebble: VirtualClone failed iterating range-del on source %s",
							m.TableNum)
					}
					if t, ok := clipAndTranslate(s); ok {
						rangedelSpans = append(rangedelSpans, t)
					}
				}
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
			if rkey != nil {
				defer rkey.Close()
				for s, err := rkey.First(); s != nil || err != nil; s, err = rkey.Next() {
					if err != nil {
						return errors.Wrapf(err,
							"pebble: VirtualClone failed iterating range-key on source %s",
							m.TableNum)
					}
					if t, ok := clipAndTranslate(s); ok {
						rangekeySpans = append(rangekeySpans, t)
					}
				}
			}
			return nil
		}); err != nil {
		return nil, 0, err
	}
	if len(rangedelSpans) == 0 && len(rangekeySpans) == 0 {
		return nil, 0, nil
	}

	tableNum := d.mu.versions.getNextTableNum()
	fileNum := base.PhysicalTableDiskFileNum(tableNum)

	writable, _, err := d.objProvider.Create(ctx, base.FileTypeTable, fileNum,
		objstorage.CreateOptions{PreferSharedStorage: false})
	if err != nil {
		return nil, 0, errors.Wrapf(err,
			"pebble: VirtualClone failed to create fragment SST object")
	}

	// Use the source level's writer options for format + block-prop collectors;
	// the actual placement is L0 (set on the clonePlanEntry).
	writerOpts := d.opts.MakeWriterOptions(level, d.TableFormat())
	tw := sstable.NewRawWriter(writable, writerOpts)
	for i := range rangedelSpans {
		if err := tw.EncodeSpan(rangedelSpans[i]); err != nil {
			_ = tw.Close()
			_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
			return nil, 0, errors.Wrapf(err, "pebble: VirtualClone fragment range-del write")
		}
	}
	for i := range rangekeySpans {
		if err := tw.EncodeSpan(rangekeySpans[i]); err != nil {
			_ = tw.Close()
			_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
			return nil, 0, errors.Wrapf(err, "pebble: VirtualClone fragment range-key write")
		}
	}
	if err := tw.Close(); err != nil {
		_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
		return nil, 0, errors.Wrapf(err, "pebble: VirtualClone fragment close")
	}
	wm, err := tw.Metadata()
	if err != nil {
		_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
		return nil, 0, errors.Wrapf(err, "pebble: VirtualClone fragment metadata")
	}

	pm := &manifest.TableMetadata{
		TableNum:              tableNum,
		Size:                  wm.Size,
		SeqNums:               wm.SeqNums,
		LargestSeqNumAbsolute: wm.SeqNums.High,
	}
	if len(rangedelSpans) > 0 {
		pm.ExtendPointKeyBounds(d.cmp, wm.SmallestRangeDel, wm.LargestRangeDel)
	}
	if len(rangekeySpans) > 0 {
		pm.ExtendRangeKeyBounds(d.cmp, manifest.AnyRangeKeys, wm.SmallestRangeKey, wm.LargestRangeKey)
	}
	pm.InitPhysicalBacking()
	if err := pm.Validate(d.cmp, d.opts.Comparer.FormatKey); err != nil {
		_ = d.objProvider.Remove(base.FileTypeTable, fileNum)
		return nil, 0, errors.Wrapf(err,
			"pebble: VirtualClone fragment table %s is invalid", pm.TableNum)
	}
	return pm, fileNum, nil
}
