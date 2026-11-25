import type { RolloutRow } from "@/lib/api";

export type Group = {
  promptId: string;
  groupIdx: number;
  prompt: string;
  samples: RolloutRow[];
  rewardMean: number;
  rewardStd: number;
  correct: number; // samples with the "correct" component at 1
  zeroAdvantage: boolean; // every sample got the same reward: no learning signal
};

export function components(row: RolloutRow): Record<string, number> {
  try {
    return JSON.parse(row.reward_components) as Record<string, number>;
  } catch {
    return {};
  }
}

export function groupRollouts(rows: RolloutRow[]): Group[] {
  const byPrompt = new Map<string, RolloutRow[]>();
  for (const row of rows) {
    const list = byPrompt.get(row.prompt_id) ?? [];
    list.push(row);
    byPrompt.set(row.prompt_id, list);
  }
  return [...byPrompt.entries()]
    .map(([promptId, list]) => {
      const samples = [...list].sort((a, b) => a.sample_idx - b.sample_idx);
      const rewards = samples.map((s) => s.reward_total);
      const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
      const std = Math.sqrt(rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / rewards.length);
      return {
        promptId,
        groupIdx: samples[0].group_idx,
        prompt: samples[0].prompt,
        samples,
        rewardMean: mean,
        rewardStd: std,
        correct: samples.filter((s) => components(s).correct === 1).length,
        zeroAdvantage: rewards.every((r) => r === rewards[0]),
      };
    })
    .sort((a, b) => a.groupIdx - b.groupIdx);
}
