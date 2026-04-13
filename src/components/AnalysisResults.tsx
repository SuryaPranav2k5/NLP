import { AnalysisResult, EntityType } from '@/types/analysis';
import { SeverityBadge } from './SeverityBadge';
import { EntityGroup } from './EntityGroup';
import { FileText, Target } from 'lucide-react';

interface AnalysisResultsProps {
  result: AnalysisResult;
}

const entityTypeOrder: EntityType[] = ['vehicle', 'location', 'violation', 'time', 'zone', 'other'];

export function AnalysisResults({ result }: AnalysisResultsProps) {
  // Group entities by type
  const groupedEntities = entityTypeOrder.reduce((acc, type) => {
    acc[type] = result.entities.filter(e => e.type === type);
    return acc;
  }, {} as Record<EntityType, typeof result.entities>);

  const hasEntities = result.entities.length > 0;

  return (
    <div className="animate-fade-in space-y-6">
      {/* Category and Severity */}
      <div className="bg-card rounded-xl border border-border p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <Target className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1">Predicted Category</p>
              <h3 className="text-xl font-semibold text-foreground">{result.category}</h3>
              {result.subtype && (
                <p className="text-sm text-muted-foreground mt-1">
                  Subtype: <span className="font-medium text-foreground">{result.subtype}</span>
                </p>
              )}
            </div>
          </div>
          <SeverityBadge severity={result.severity} />
        </div>
      </div>

      {/* Extracted Entities */}
      <div className="bg-card rounded-xl border border-border p-6 shadow-sm">
        <div className="flex items-center gap-2 mb-5">
          <FileText className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold text-foreground">Extracted Entities</h3>
        </div>

        {hasEntities ? (
          <div className="grid gap-5 sm:grid-cols-2">
            {entityTypeOrder.map(type => (
              <EntityGroup
                key={type}
                type={type}
                entities={groupedEntities[type]}
              />
            ))}
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">
            No specific entities were extracted from this note.
          </p>
        )}
      </div>
    </div>
  );
}
