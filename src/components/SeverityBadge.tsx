import { SeverityLevel } from '@/types/analysis';
import { AlertCircle, AlertTriangle, Info } from 'lucide-react';

interface SeverityBadgeProps {
  severity: SeverityLevel;
}

const severityConfig: Record<SeverityLevel, { label: string; icon: React.ElementType; className: string }> = {
  low: { label: 'Low Severity', icon: Info, className: 'severity-low' },
  medium: { label: 'Medium Severity', icon: AlertTriangle, className: 'severity-medium' },
  high: { label: 'High Severity', icon: AlertCircle, className: 'severity-high' },
};

export function SeverityBadge({ severity }: SeverityBadgeProps) {
  const config = severityConfig[severity];
  const Icon = config.icon;

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${config.className}`}>
      <Icon className="h-4 w-4" />
      {config.label}
    </span>
  );
}
