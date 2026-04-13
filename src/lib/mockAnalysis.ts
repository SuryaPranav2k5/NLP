import { AnalysisResult, Entity, EntityType, SeverityLevel } from '@/types/analysis';

// Mock analysis function - simulates API call
export async function analyzePatrolNote(text: string): Promise<AnalysisResult> {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 1500));

  const textLower = text.toLowerCase();

  // Determine category based on keywords
  let category = 'General Parking Violation';
  if (textLower.includes('meter') || textLower.includes('expired')) {
    category = 'Expired Meter';
  } else if (textLower.includes('hydrant') || textLower.includes('fire')) {
    category = 'Fire Hydrant Zone';
  } else if (textLower.includes('handicap') || textLower.includes('disabled') || textLower.includes('accessible')) {
    category = 'Accessible Parking Violation';
  } else if (textLower.includes('double park') || textLower.includes('double-park')) {
    category = 'Double Parking';
  } else if (textLower.includes('no parking') || textLower.includes('no-parking')) {
    category = 'No Parking Zone';
  } else if (textLower.includes('loading') || textLower.includes('commercial')) {
    category = 'Loading Zone Violation';
  } else if (textLower.includes('overnight') || textLower.includes('street cleaning')) {
    category = 'Street Cleaning Violation';
  }

  // Determine severity
  let severity: SeverityLevel = 'low';
  if (textLower.includes('hydrant') || textLower.includes('handicap') || textLower.includes('disabled') || textLower.includes('blocking')) {
    severity = 'high';
  } else if (textLower.includes('double') || textLower.includes('no parking')) {
    severity = 'medium';
  }

  // Extract entities using simple pattern matching
  const entities: Entity[] = [];

  // Vehicle patterns
  const vehiclePatterns = [
    /(?:vehicle|car|truck|van|suv|sedan|coupe)\s*(?:plate|license|#|number)?:?\s*([A-Z0-9]{2,8})/gi,
    /(?:plate|license)\s*(?:#|number)?:?\s*([A-Z0-9]{2,8})/gi,
    /([A-Z]{2,3}[\s-]?[0-9]{3,4})/g,
    /(blue|red|white|black|silver|gray|grey|green|yellow|orange|brown)\s+(sedan|suv|truck|van|car|coupe|hatchback)/gi,
  ];

  vehiclePatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => {
        if (!entities.find(e => e.text.toLowerCase() === match.toLowerCase() && e.type === 'vehicle')) {
          entities.push({ text: match.trim(), type: 'vehicle' });
        }
      });
    }
  });

  // Location patterns
  const locationPatterns = [
    /\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Ln|Lane|Way|Ct|Court)/gi,
    /(?:corner of|intersection of|near|at)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/gi,
    /(?:block|lot)\s*#?\s*\d+/gi,
  ];

  locationPatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => {
        if (!entities.find(e => e.text.toLowerCase() === match.toLowerCase() && e.type === 'location')) {
          entities.push({ text: match.trim(), type: 'location' });
        }
      });
    }
  });

  // Time patterns
  const timePatterns = [
    /\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?/g,
    /\d{1,2}\s*(?:am|pm|AM|PM)/g,
    /(?:morning|afternoon|evening|night)/gi,
    /\d{1,2}\/\d{1,2}\/\d{2,4}/g,
  ];

  timePatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => {
        if (!entities.find(e => e.text.toLowerCase() === match.toLowerCase() && e.type === 'time')) {
          entities.push({ text: match.trim(), type: 'time' });
        }
      });
    }
  });

  // Zone patterns
  const zonePatterns = [
    /zone\s*[A-Z0-9]+/gi,
    /(?:residential|commercial|loading|no[\s-]?parking)\s+zone/gi,
    /meter\s*#?\s*\d+/gi,
  ];

  zonePatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => {
        if (!entities.find(e => e.text.toLowerCase() === match.toLowerCase() && e.type === 'zone')) {
          entities.push({ text: match.trim(), type: 'zone' });
        }
      });
    }
  });

  // Violation object patterns
  const violationPatterns = [
    /(?:fire\s+)?hydrant/gi,
    /(?:expired|broken)\s+meter/gi,
    /(?:no\s+)?parking\s+(?:sign|meter)/gi,
    /(?:handicap|accessible|disabled)\s+(?:spot|space|parking)/gi,
    /crosswalk/gi,
    /sidewalk/gi,
    /driveway/gi,
  ];

  violationPatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => {
        if (!entities.find(e => e.text.toLowerCase() === match.toLowerCase() && e.type === 'violation')) {
          entities.push({ text: match.trim(), type: 'violation' });
        }
      });
    }
  });

  // Calculate mock confidence
  const confidence = Math.min(0.95, 0.7 + (entities.length * 0.05));

  return {
    category,
    severity,
    entities,
    confidence,
  };
}
