import { Entity, EntityType } from '@/types/analysis';
import { Car, MapPin, AlertTriangle, Clock, Map, Tag } from 'lucide-react';

interface EntityGroupProps {
  type: EntityType;
  entities: Entity[];
}

const entityConfig: Record<EntityType, { label: string; icon: React.ElementType; className: string }> = {
  vehicle: { label: 'Vehicle', icon: Car, className: 'entity-vehicle' },
  location: { label: 'Location', icon: MapPin, className: 'entity-location' },
  violation: { label: 'Violation Object', icon: AlertTriangle, className: 'entity-violation' },
  time: { label: 'Time', icon: Clock, className: 'entity-time' },
  zone: { label: 'Zone', icon: Map, className: 'entity-zone' },
  other: { label: 'Other', icon: Tag, className: 'entity-other' },
};

export function EntityGroup({ type, entities }: EntityGroupProps) {
  const config = entityConfig[type];
  const Icon = config.icon;

  if (entities.length === 0) return null;

  return (
    <div className="animate-fade-in">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="h-4 w-4 text-muted-foreground" />
        <h4 className="text-sm font-medium text-muted-foreground">{config.label}</h4>
      </div>
      <div className="flex flex-wrap gap-2">
        {entities.map((entity, index) => (
          <span
            key={`${entity.text}-${index}`}
            className={`entity-tag ${config.className}`}
          >
            {entity.text}
          </span>
        ))}
      </div>
    </div>
  );
}
