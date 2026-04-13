export type SeverityLevel = 'low' | 'medium' | 'high';

export type EntityType = 'vehicle' | 'location' | 'violation' | 'time' | 'zone' | 'other';

export interface Entity {
  text: string;
  type: EntityType;
}

export interface AnalysisResult {
  category: string;
  subtype: string;
  severity: SeverityLevel;
  entities: Entity[];
  inferenceTimeMs?: number;
}
