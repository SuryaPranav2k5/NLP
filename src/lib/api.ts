import { AnalysisResult } from '@/types/analysis';

const API_BASE_URL = 'http://localhost:8000';

export async function analyzePatrolNote(text: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  const data = await response.json();

  // Transform backend response to match frontend AnalysisResult type
  return {
    category: data.violation_category,
    subtype: data.violation_subtype,
    severity: data.severity.toLowerCase() as 'low' | 'medium' | 'high',
    entities: transformEntities(data.entities),
    inferenceTimeMs: data.inference_time_ms,
  };
}

// Transform backend entity format to frontend format
function transformEntities(backendEntities: Record<string, string[]>) {
  const entities: AnalysisResult['entities'] = [];

  // Map backend entity types to frontend entity types
  const entityTypeMap: Record<string, 'vehicle' | 'location' | 'violation' | 'time' | 'zone' | 'other'> = {
    'LOCATION': 'location',
    'VEHICLE_COLOR': 'vehicle',
    'VEHICLE_TYPE': 'vehicle',
    'TIME_RANGE': 'time',
    'ZONE_TYPE': 'zone',
    'VIOLATION_OBJECT': 'violation',
    'PERMIT_STATUS': 'violation',
    'REPEAT_STATUS': 'other',
  };

  for (const [entityType, values] of Object.entries(backendEntities)) {
    const frontendType = entityTypeMap[entityType] || 'other';

    values.forEach(value => {
      entities.push({
        text: value,
        type: frontendType,
      });
    });
  }

  return entities;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    const data = await response.json();
    return data.status === 'running';
  } catch {
    return false;
  }
}
