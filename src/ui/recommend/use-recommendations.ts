import { useEffect, useSyncExternalStore } from 'react';
import { recommendationStore, type RecommendationSnapshot } from '../../recommend/recommendation-store';
import type { Recommendation } from '../../recommend/recommend';

export function useRecommendations(): RecommendationSnapshot {
  const snapshot = useSyncExternalStore(recommendationStore.subscribe, recommendationStore.getSnapshot);
  useEffect(() => {
    void recommendationStore.init();
  }, []);
  return snapshot;
}

export function useRecommendation(modelId: string): Recommendation | undefined {
  return useRecommendations().recommendations.find((r) => r.model.id === modelId);
}
