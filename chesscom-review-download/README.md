# Chess.com Game Review Scraper

Bookmarklet that extracts move-by-move review data from chess.com into CSV files
for the ChessLens classification optimizer.

## Install

1. Create a new bookmark in your browser (right-click bookmarks bar → **Add page**
   or **Add bookmark**)
2. Set the name to anything you like (e.g. `Extract Review`)
3. Paste the entire contents of `bookmarklet.min.txt` as the URL
4. Save the bookmark

## Use

1. Open a completed **Game Review** on chess.com
2. Wait for the full move list with classifications to load
3. Click the bookmark — a CSV file downloads automatically

The CSV is named `game_<id>.csv` (using the game ID from the URL) and is ready to
drop into `tests/test_games/`.

## CSV columns

| Column | Example | Notes |
|---|---|---|
| `ply` | 1, 2, 3… | 1-indexed. White = odd, Black = even |
| `color` | white / black | |
| `move` | e4, Nf3, O-O | SAN notation |
| `classification` | best, blunder… | book, best, excellent, good, inaccuracy, mistake, blunder, great, brilliant, miss |
| `eval_points` | 0.45, -1.2, M5 | White's POV. `M5` = mate in 5, `1-0` / `0-1` / `1/2-1/2` for game end |
| `points_difference` | -0.45, 1.2 | Eval change from previous position (blank for game end) |

## Rebuild after editing

If you edit `bookmarklet.js` (e.g. to update selectors), regenerate the
paste-ready bookmarklet:

```bash
node -e "
var fs = require('fs');
var src = fs.readFileSync('bookmarklet.js', 'utf8');
var m = src
  .replace(/\/\/[^\n]*/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/\s+/g, ' ')
  .replace(/\s*([{}();,=+\-*\/<>!&|?:])\s*/g, '\$1')
  .replace(/^\s+|\s+$/g, '');
fs.writeFileSync('bookmarklet.min.txt', 'javascript:' + encodeURIComponent(m));
console.log('Written bookmarklet.min.txt (' + m.length + ' chars)');
"
```

Then re-paste the contents of `bookmarklet.min.txt` into your bookmark URL.

## Troubleshooting

- **Empty CSV / "Could not extract moves"** — chess.com may have changed its DOM.
  Open DevTools on a reviewed game, inspect a move row, and update the `SELECTORS`
  object at the top of `bookmarklet.js` to match.
- **Download doesn't trigger** — paste the contents of `bookmarklet.min.txt` into
  the browser console on the chess.com page instead.
- **`chesslensDebug()`** — if extraction fails, this function is exposed on
  `window`. Run it in the console for diagnostic info about the page's DOM.
