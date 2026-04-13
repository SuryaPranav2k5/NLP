import { FileSearch } from 'lucide-react';

export function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="p-4 rounded-full bg-muted mb-4">
        <FileSearch className="h-8 w-8 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-medium text-foreground mb-2">No Analysis Yet</h3>
      <p className="text-muted-foreground text-sm max-w-sm">
        Enter a patrol note in the text area above and click "Analyze" to extract violation details and categorize the note.
      </p>
    </div>
  );
}
