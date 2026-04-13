import { Loader2 } from 'lucide-react';

export function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="p-4 rounded-full bg-primary/10 mb-4">
        <Loader2 className="h-8 w-8 text-primary animate-spin" />
      </div>
      <h3 className="text-lg font-medium text-foreground mb-2">Analyzing Note</h3>
      <p className="text-muted-foreground text-sm">
        Processing your patrol note...
      </p>
    </div>
  );
}
