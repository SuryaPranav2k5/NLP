import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { AnalysisResults } from '@/components/AnalysisResults';
import { EmptyState } from '@/components/EmptyState';
import { LoadingState } from '@/components/LoadingState';
import { analyzePatrolNote } from '@/lib/api';
import { AnalysisResult } from '@/types/analysis';
import { Search, RotateCcw, FileText } from 'lucide-react';
import { toast } from 'sonner';

const MAX_CHARS = 2000;

const Index = () => {
  const [patrolNote, setPatrolNote] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleAnalyze = async () => {
    if (!patrolNote.trim()) {
      toast.error('Please enter a patrol note to analyze');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const analysisResult = await analyzePatrolNote(patrolNote);
      setResult(analysisResult);
      toast.success('Analysis complete');
    } catch (error) {
      toast.error('Failed to analyze the note. Please try again.');
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClear = () => {
    setPatrolNote('');
    setResult(null);
  };

  const charCount = patrolNote.length;
  const isOverLimit = charCount > MAX_CHARS;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container max-w-4xl py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <FileText className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-foreground">
                Parking Violation Text Analytics
              </h1>
              <p className="text-muted-foreground text-sm mt-0.5">
                Analyze municipal parking patrol notes to extract violations, entities, and severity levels
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container max-w-4xl py-8">
        <div className="space-y-8">
          {/* Input Section */}
          <section className="bg-card rounded-xl border border-border p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <label htmlFor="patrol-note" className="text-sm font-medium text-foreground">
                Patrol Note
              </label>
              <span className={`text-xs ${isOverLimit ? 'text-destructive font-medium' : 'text-muted-foreground'}`}>
                {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()} characters
              </span>
            </div>
            
            <Textarea
              id="patrol-note"
              placeholder="Paste or type a patrol note here...

Example: White sedan ABC-1234 parked at fire hydrant on 123 Main Street. Meter #456 expired at 2:30 PM. Zone C residential area."
              value={patrolNote}
              onChange={(e) => setPatrolNote(e.target.value)}
              className="min-h-[160px] resize-y text-base leading-relaxed"
              disabled={isAnalyzing}
            />

            <div className="flex items-center gap-3 mt-4">
              <Button 
                onClick={handleAnalyze} 
                disabled={isAnalyzing || !patrolNote.trim() || isOverLimit}
                className="gap-2"
              >
                <Search className="h-4 w-4" />
                Analyze
              </Button>
              
              {(patrolNote || result) && (
                <Button 
                  variant="outline" 
                  onClick={handleClear}
                  disabled={isAnalyzing}
                  className="gap-2"
                >
                  <RotateCcw className="h-4 w-4" />
                  Clear
                </Button>
              )}
            </div>
          </section>

          {/* Results Section */}
          <section>
            {isAnalyzing ? (
              <LoadingState />
            ) : result ? (
              <AnalysisResults result={result} />
            ) : (
              <EmptyState />
            )}
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-card mt-auto">
        <div className="container max-w-4xl py-4">
          <p className="text-xs text-muted-foreground text-center">
            Parking Violation Text Analytics — Municipal Data Processing Tool
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
