/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { describe, expect, it } from "vitest";
import {
	appendOnlineState,
	initialOnlineState,
	ONLINE_STATE_ENTRY,
	recordBoundary,
	recordCompaction,
	recordCompletedPlanHandoff,
	recordCorrection,
	recordProviderRequest,
	restoreOnlineState,
} from "../src/sol-pi/extensions/online-context-compact/state.ts";
import { analyzePlanTransition } from "../src/sol-pi/extensions/online-context-compact/plan.ts";
import { FakePi, FakeSessionManager } from "./helpers.ts";

const PLAN = [
	{ id: "inspect", goal: "inspect the implementation", status: "completed" as const },
	{ id: "verify", goal: "verify the change", status: "in_progress" as const },
];

const PROGRESS = {
	stepId: "inspect",
	goal: "inspect the implementation",
	filesChanged: ["src/a.ts"],
	verification: ["targeted test passed"],
	decisions: ["keep the change small"],
	nextWork: ["verify the change"],
};

describe("Online Context Compact state snapshots", () => {
	it("starts with a disabled-by-default empty state", () => {
		expect(initialOnlineState()).toEqual({
			version: 1,
			epoch: 0,
			plan: [],
			awaitingPlanRestatement: false,
			pendingProgress: [],
			requestCount: 0,
			lastCompactionRequestCount: null,
			lastBoundaryRequestCount: 0,
			completedBoundaryRequestCounts: [],
			lastContextTokens: null,
			positiveContextDeltaTotal: 0,
			positiveContextDeltaCount: 0,
			nativeCompactionCount: 0,
			cacheDebtTokens: 0,
			cacheDebtRepaymentTokens: 0,
		});
	});

	it("restores the latest valid snapshot and ignores a malformed tail", () => {
		const manager = new FakeSessionManager();
		const pi = new FakePi(manager);
		const state = recordBoundary(recordProviderRequest(initialOnlineState(), 100), PLAN, PROGRESS);
		appendOnlineState(pi.asExtensionApi(), state);
		manager.appendCustomEntry(ONLINE_STATE_ENTRY, { version: 1, plan: "broken" });

		expect(restoreOnlineState(manager.entries)).toEqual(state);
	});

	it("resets an oversized legacy plan without rolling back the latest accounting", () => {
		const manager = new FakeSessionManager();
		const pi = new FakePi(manager);
		const legacy = {
			...recordProviderRequest(initialOnlineState(), 1_000),
			plan: [{ id: "legacy", goal: "g".repeat(1_001), status: "in_progress" as const }],
			cacheDebtTokens: 900,
			cacheDebtRepaymentTokens: 300,
		};
		appendOnlineState(pi.asExtensionApi(), legacy);

		expect(restoreOnlineState(manager.entries)).toEqual({ ...legacy, plan: [] });
	});

	it("counts requests, positive context growth, and cache-debt repayment", () => {
		const charged = {
			...initialOnlineState(),
			cacheDebtTokens: 300,
			cacheDebtRepaymentTokens: 100,
		};
		const first = recordProviderRequest(charged, 1_000);
		const second = recordProviderRequest(first, 1_250);
		const third = recordProviderRequest(second, 900);

		expect(third).toMatchObject({
			requestCount: 3,
			lastContextTokens: 900,
			positiveContextDeltaTotal: 250,
			positiveContextDeltaCount: 1,
			cacheDebtTokens: 0,
			cacheDebtRepaymentTokens: 0,
		});
	});

	it("records one request interval and one progress summary per boundary", () => {
		let state = initialOnlineState();
		state = recordProviderRequest(state, 100);
		state = recordProviderRequest(state, 200);
		state = recordBoundary(state, PLAN, PROGRESS);
		state = recordProviderRequest(state, 300);
		state = recordBoundary(state, PLAN, undefined);

		expect(state.completedBoundaryRequestCounts).toEqual([2, 1]);
		expect(state.lastBoundaryRequestCount).toBe(3);
		expect(state.pendingProgress).toEqual([PROGRESS]);
	});

	it("starts a clean epoch after native compaction and carries its cache debt", () => {
		const before = recordBoundary(recordProviderRequest(initialOnlineState(), 5_000), PLAN, PROGRESS);
		const after = recordCompaction(before, { debtTokens: 1_200, repaymentTokens: 300 });

		expect(after).toMatchObject({
			epoch: 1,
			plan: PLAN,
			awaitingPlanRestatement: true,
			lastCompactionRequestCount: before.requestCount,
			pendingProgress: [],
			nativeCompactionCount: 1,
			cacheDebtTokens: 1_200,
			cacheDebtRepaymentTokens: 300,
		});
	});

	it("resets boundary cadence after compaction without losing plan restatement or debt", () => {
		const before = recordBoundary(recordProviderRequest(initialOnlineState(), 5_000), PLAN, PROGRESS);
		const after = recordCompaction(before, { debtTokens: 1_200, repaymentTokens: 300 });
		expect(after).toMatchObject({
			plan: PLAN,
			awaitingPlanRestatement: true,
			lastBoundaryRequestCount: 1,
			completedBoundaryRequestCounts: [],
			cacheDebtTokens: 1_200,
		});
		expect(recordBoundary(recordProviderRequest(after, 5_100), PLAN, undefined).completedBoundaryRequestCounts).toEqual([1]);
	});

	it("starts fresh horizon only for a completed-plan follow-up", () => {
		const completed = [{ id: "done", goal: "finish", status: "completed" as const }];
		const before = recordBoundary(recordProviderRequest(initialOnlineState(), 5_000), completed, PROGRESS);
		const after = recordCompletedPlanHandoff(before);
		expect(after).toMatchObject({
			epoch: 1,
			plan: [],
			completedBoundaryRequestCounts: [],
			lastBoundaryRequestCount: 1,
			cacheDebtTokens: 0,
		});
		expect(recordCompletedPlanHandoff(after)).toBe(after);
		expect(recordCompletedPlanHandoff(recordBoundary(initialOnlineState(), PLAN, undefined))).toMatchObject({ plan: PLAN });
	});

	it("accumulates unpaid debt and savings across successive compactions", () => {
		const outstanding = {
			...initialOnlineState(),
			cacheDebtTokens: 900,
			cacheDebtRepaymentTokens: 300,
		};
		const after = recordCompaction(outstanding, { debtTokens: 800, repaymentTokens: 200 });

		expect(after).toMatchObject({
			cacheDebtTokens: 1_700,
			cacheDebtRepaymentTokens: 500,
		});
		expect(recordCompaction(outstanding, { debtTokens: 0, repaymentTokens: 0 })).toMatchObject({
			cacheDebtTokens: 900,
			cacheDebtRepaymentTokens: 300,
		});
	});

	it("drops stale plan history and debt when the user corrects an active run", () => {
		const before = {
			...recordBoundary(recordProviderRequest(initialOnlineState(), 5_000), PLAN, PROGRESS),
			cacheDebtTokens: 900,
			cacheDebtRepaymentTokens: 300,
		};
		expect(recordCorrection(before)).toMatchObject({
			epoch: 1,
			plan: [],
			pendingProgress: [],
			completedBoundaryRequestCounts: [],
			lastContextTokens: null,
			cacheDebtTokens: 0,
			cacheDebtRepaymentTokens: 0,
		});
	});

	it("arms the restatement flag again after a correction", () => {
		const before = recordCompaction(
			recordBoundary(recordProviderRequest(initialOnlineState(), 5_000), PLAN, PROGRESS),
			{ debtTokens: 0, repaymentTokens: 0 },
		);
		const accepted = { ...before, awaitingPlanRestatement: false };
		expect(recordCorrection(accepted)).toMatchObject({
			awaitingPlanRestatement: true,
			plan: [],
		});
	});

	it("restores snapshots persisted before the restatement bookkeeping existed", () => {
		const legacy = initialOnlineState();
		const persisted = JSON.parse(JSON.stringify(legacy)) as Record<string, unknown>;
		delete persisted.awaitingPlanRestatement;
		delete persisted.lastCompactionRequestCount;

		const manager = new FakeSessionManager();
		manager.appendCustomEntry(ONLINE_STATE_ENTRY, persisted);

		expect(restoreOnlineState(manager.entries)).toEqual(legacy);
	});

	it("does not treat a plan re-issued after compaction as new progress", () => {
		const before = recordBoundary(recordProviderRequest(initialOnlineState(), 5_000), PLAN, PROGRESS);
		const after = recordCompaction(before, { debtTokens: 0, repaymentTokens: 0 });

		expect(analyzePlanTransition(after.plan, PLAN).completedSteps).toEqual([]);
	});

});
