/**
 * Utility functions for text selection handling
 */

class TextSelectionManager {
  constructor() {
    this.selectedText = '';
    this.onSelectionChange = null;

    // Only initialize if running in browser environment
    if (typeof window !== 'undefined' && typeof document !== 'undefined') {
      this.init();
    }
  }

  init() {
    // Listen for selection changes
    document.addEventListener('selectionchange', this.handleSelectionChange.bind(this));

    // Also listen for mouseup and keyup for broader compatibility
    document.addEventListener('mouseup', this.handleSelectionChange.bind(this));
    document.addEventListener('keyup', this.handleSelectionChange.bind(this));
  }

  handleSelectionChange() {
    // Only run if in browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    // Only trigger if there's actual selected text
    if (selectedText && selectedText !== this.selectedText) {
      this.selectedText = selectedText;

      // Call the callback if registered
      if (this.onSelectionChange) {
        this.onSelectionChange(selectedText);
      }
    } else if (!selectedText && this.selectedText) {
      // Clear the selection if it was deselected
      this.selectedText = '';
      if (this.onSelectionChange) {
        this.onSelectionChange('');
      }
    }
  }

  /**
   * Get currently selected text
   */
  getSelectedText() {
    return this.selectedText;
  }

  /**
   * Register a callback for selection changes
   */
  onSelection(callback) {
    this.onSelectionChange = callback;
  }

  /**
   * Get selection context (surrounding text, element info)
   */
  getSelectionContext() {
    // Only run if in browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return null;
    }

    const selection = window.getSelection();
    if (!selection.rangeCount) return null;

    const range = selection.getRangeAt(0);
    const selectedElement = range.commonAncestorContainer;

    // Get surrounding text for context
    const startOffset = Math.max(0, range.startOffset - 50);
    const endOffset = Math.min(range.endContainer.textContent?.length || 0, range.endOffset + 50);

    let contextBefore = '';
    let contextAfter = '';

    if (range.startContainer.textContent) {
      contextBefore = range.startContainer.textContent.substring(startOffset, range.startOffset).slice(-50);
      contextAfter = range.endContainer.textContent.substring(range.endOffset, endOffset).substring(0, 50);
    }

    return {
      text: selection.toString(),
      element: selectedElement,
      contextBefore,
      contextAfter,
      rect: range.getBoundingClientRect ? range.getBoundingClientRect() : { left: 0, top: 0, width: 0, height: 0 },
      elementInfo: {
        tagName: selectedElement.parentElement?.tagName || selectedElement.tagName || 'TEXT_NODE',
        className: selectedElement.parentElement?.className || selectedElement.className || '',
        id: selectedElement.parentElement?.id || selectedElement.id || ''
      }
    };
  }

  /**
   * Clear current selection
   */
  clearSelection() {
    // Only run if in browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      this.selectedText = '';
      return;
    }

    if (window.getSelection) {
      window.getSelection().removeAllRanges();
    } else if (document.selection) {
      document.selection.empty();
    }
    this.selectedText = '';
  }

  /**
   * Validate if selection is appropriate for context
   */
  isValidSelection(selectionText) {
    // Filter out selections that are too short or likely not useful
    if (selectionText.length < 5) return false;

    // Filter out selections with mostly whitespace
    if (selectionText.replace(/\s/g, '').length < 3) return false;

    // Filter out selections that look like they might be navigation elements
    const trimmed = selectionText.trim();
    if (trimmed.length > 1000) return false; // Too long

    return true;
  }

  /**
   * Get formatted context from selection
   */
  getFormattedContext() {
    const context = this.getSelectionContext();
    if (!context || !this.isValidSelection(context.text)) {
      return null;
    }

    // Only access scrollX and scrollY if in browser environment
    const scrollX = typeof window !== 'undefined' ? window.scrollX : 0;
    const scrollY = typeof window !== 'undefined' ? window.scrollY : 0;

    return {
      selectedText: context.text,
      sourceElement: context.elementInfo,
      surroundingContext: {
        before: context.contextBefore,
        after: context.contextAfter
      },
      position: {
        x: context.rect.left + scrollX,
        y: context.rect.top + scrollY,
        width: context.rect.width,
        height: context.rect.height
      }
    };
  }
}

// Create a singleton instance
const textSelectionManager = new TextSelectionManager();

export default textSelectionManager;