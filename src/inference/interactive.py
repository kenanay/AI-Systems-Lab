"""
src/inference/interactive.py

Interactive Text Generation Interface

Bu modül interactive text generation için CLI interface sağlar:
- REPL (Read-Eval-Print Loop) for text generation
- Context management (conversation history)
- Multi-turn generation
- Real-time streaming output
- Command system (/help, /clear, /config, etc.)

Interactive mode kullanıcının model ile doğal dil etkileşimini sağlar.

Kullanım:
    python -m src.inference.interactive \\
        --model checkpoints/model.pt \\
        --tokenizer tokenizer.json
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging

from src.inference.pipeline import InferencePipeline
from src.inference.config import GenerationConfig

logger = logging.getLogger(__name__)


class InteractiveSession:
    """
    Interactive text generation session.
    
    Features:
    - Multi-turn conversation
    - Context management (tracks history)
    - Streaming output
    - Command system
    - Configurable generation parameters
    
    Commands:
        /help - Show help message
        /clear - Clear conversation history
        /config - Show current config
        /temp <value> - Set temperature
        /topk <value> - Set top-k
        /topp <value> - Set top-p
        /tokens <value> - Set max_new_tokens
        /quit or /exit - Exit session
    
    Args:
        pipeline: Inference pipeline
        config: Initial generation config
        max_context_length: Maximum tokens to keep in context
        show_streaming: Show tokens as they are generated
    
    Example:
        >>> pipeline = InferencePipeline.from_pretrained(
        ...     "model.pt", "tokenizer.json"
        ... )
        >>> session = InteractiveSession(pipeline)
        >>> session.run()
    """
    
    def __init__(
        self,
        pipeline: InferencePipeline,
        config: Optional[GenerationConfig] = None,
        max_context_length: int = 1024,
        show_streaming: bool = True
    ):
        """Initialize interactive session."""
        self.pipeline = pipeline
        self.config = config or GenerationConfig.sampling()
        self.max_context_length = max_context_length
        self.show_streaming = show_streaming
        
        # Conversation history
        self.context: List[str] = []
        
        logger.info("Interactive session initialized")
    
    def run(self):
        """
        Run interactive REPL loop.
        
        Reads user input, generates response, maintains context.
        """
        print("=" * 80)
        print("Interactive Text Generation")
        print("=" * 80)
        print("\nType /help for commands, /quit to exit\n")
        
        while True:
            try:
                # Get user input
                prompt = input("\n>>> ").strip()
                
                if not prompt:
                    continue
                
                # Check for commands
                if prompt.startswith('/'):
                    should_continue = self._handle_command(prompt)
                    if not should_continue:
                        break
                    continue
                
                # Generate response
                self._generate_and_display(prompt)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
                continue
            except EOFError:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.exception("Error in interactive session")
    
    def _generate_and_display(self, prompt: str):
        """
        Generate response and display it.
        
        Args:
            prompt: User input
        """
        # Add to context
        self.context.append(prompt)
        
        # Build full prompt with context
        full_prompt = self._build_context_prompt()
        
        print("\n", end="", flush=True)
        
        if self.show_streaming:
            # Stream generation
            for token_text in self.pipeline.generate_stream(
                full_prompt,
                config=self.config
            ):
                print(token_text, end="", flush=True)
            print()  # Newline at end
        else:
            # Non-streaming generation
            response = self.pipeline(full_prompt, config=self.config)
            print(response)
        
        # Note: In real implementation, we'd extract just the new text
        # For now, showing full output
    
    def _build_context_prompt(self) -> str:
        """
        Build prompt with conversation history.
        
        Returns:
            Full prompt string with context
        """
        # Simple concatenation for now
        # In practice, you'd want to format properly and manage length
        return " ".join(self.context[-5:])  # Keep last 5 turns
    
    def _handle_command(self, command: str) -> bool:
        """
        Handle user command.
        
        Args:
            command: Command string (starts with /)
        
        Returns:
            True to continue session, False to exit
        """
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in ['/quit', '/exit', '/q']:
            print("\nGoodbye!")
            return False
        
        elif cmd == '/help':
            self._show_help()
        
        elif cmd == '/clear':
            self.context.clear()
            print("✓ Context cleared")
        
        elif cmd == '/config':
            self._show_config()
        
        elif cmd == '/temp':
            if args:
                try:
                    self.config.temperature = float(args[0])
                    print(f"✓ Temperature set to {self.config.temperature}")
                except ValueError:
                    print("❌ Invalid temperature value")
            else:
                print(f"Current temperature: {self.config.temperature}")
        
        elif cmd == '/topk':
            if args:
                try:
                    self.config.top_k = int(args[0])
                    print(f"✓ Top-k set to {self.config.top_k}")
                except ValueError:
                    print("❌ Invalid top-k value")
            else:
                print(f"Current top-k: {self.config.top_k}")
        
        elif cmd == '/topp':
            if args:
                try:
                    self.config.top_p = float(args[0])
                    print(f"✓ Top-p set to {self.config.top_p}")
                except ValueError:
                    print("❌ Invalid top-p value")
            else:
                print(f"Current top-p: {self.config.top_p}")
        
        elif cmd == '/tokens':
            if args:
                try:
                    self.config.max_new_tokens = int(args[0])
                    print(f"✓ Max tokens set to {self.config.max_new_tokens}")
                except ValueError:
                    print("❌ Invalid token count")
            else:
                print(f"Current max_new_tokens: {self.config.max_new_tokens}")
        
        elif cmd == '/stream':
            self.show_streaming = not self.show_streaming
            status = "enabled" if self.show_streaming else "disabled"
            print(f"✓ Streaming {status}")
        
        elif cmd == '/context':
            if self.context:
                print(f"\nContext ({len(self.context)} turns):")
                for i, turn in enumerate(self.context, 1):
                    preview = turn[:60] + "..." if len(turn) > 60 else turn
                    print(f"  {i}. {preview}")
            else:
                print("Context is empty")
        
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type /help for available commands")
        
        return True
    
    def _show_help(self):
        """Show help message."""
        print("""
Commands:
  /help              - Show this help message
  /quit, /exit, /q   - Exit interactive session
  /clear             - Clear conversation history
  /context           - Show current context
  /config            - Show current generation config
  
  /temp <value>      - Set temperature (e.g., /temp 0.8)
  /topk <value>      - Set top-k (e.g., /topk 50)
  /topp <value>      - Set top-p (e.g., /topp 0.9)
  /tokens <value>    - Set max_new_tokens (e.g., /tokens 100)
  /stream            - Toggle streaming output

Examples:
  >>> Bir varmış bir yokmuş
  >>> /temp 1.2
  >>> /tokens 200
  >>> Once upon a time
        """)
    
    def _show_config(self):
        """Show current config."""
        print(f"""
Current Configuration:
  Strategy:      {self.config.strategy.value}
  Temperature:   {self.config.temperature}
  Top-k:         {self.config.top_k}
  Top-p:         {self.config.top_p}
  Max tokens:    {self.config.max_new_tokens}
  Repetition:    {self.config.repetition_penalty}
  Streaming:     {self.show_streaming}
        """)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Text Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    
    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='Path to tokenizer file (.json)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu, cuda, mps). Auto-detect if not specified.'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling (default: 50)'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p (nucleus) sampling (default: 0.9)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate (default: 100)'
    )
    
    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Disable streaming output'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Load pipeline
    print("Loading model and tokenizer...")
    
    try:
        pipeline = InferencePipeline.from_pretrained(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Create config
    config = GenerationConfig.sampling(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens
    )
    
    # Start interactive session
    session = InteractiveSession(
        pipeline=pipeline,
        config=config,
        show_streaming=not args.no_stream
    )
    
    session.run()


if __name__ == "__main__":
    main()
