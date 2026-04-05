(function() {
  'use strict';

  function showStatus(msg, isError) {
    var existing = document.getElementById('chesslens-status');
    if (existing) existing.remove();
    var div = document.createElement('div');
    div.id = 'chesslens-status';
    div.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:99999;' +
      'background:' + (isError ? '#3a1010' : '#1a3a23') + ';' +
      'color:' + (isError ? '#ef4444' : '#4ade80') + ';' +
      'border:1px solid ' + (isError ? '#ef4444' : '#4ade80') + ';' +
      'border-radius:8px;padding:14px 20px;font-family:monospace;font-size:13px;' +
      'max-width:400px;box-shadow:0 4px 20px rgba(0,0,0,0.5);transition:opacity 0.3s';
    div.textContent = msg;
    document.body.appendChild(div);
    if (!isError) {
      setTimeout(function() {
        div.style.opacity = '0';
        setTimeout(function() { div.remove(); }, 300);
      }, 5000);
    }
  }

  function parseFeedback(rawText) {
    if (!rawText) return { classification: '', eval_points: '' };

    var text = rawText
      .replace(/\.cls-\d+\{[^}]*\}/g, '')
      .replace(/\s*(show\s+follow-up|show\s+free\s+piece|show\s+missed\s+\S+|show\s+pin|show\s+idea\S*|retry)\s*/gi, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    var classification = '';
    var patterns = [
      [/\bgreat\s*(move|find)?\b/i, 'great'],
      [/\bbrilliant\b/i, 'brilliant'],
      [/\bblunder\b/i, 'blunder'],
      [/\bmistake\b/i, 'mistake'],
      [/\binaccuracy\b/i, 'inaccuracy'],
      [/\bmiss(ed)?(\s+win)?\b/i, 'miss'],
      [/\bbest\b/i, 'best'],
      [/\bexcellent\b/i, 'excellent'],
      [/\bgood\b/i, 'good'],
      [/\bbook\b/i, 'book'],
    ];
    for (var i = 0; i < patterns.length; i++) {
      if (patterns[i][0].test(text)) { classification = patterns[i][1]; break; }
    }

    var evalPts = '';

    if (/\b1-0\b/.test(text)) { evalPts = '1-0'; }
    else if (/\b0-1\b/.test(text)) { evalPts = '0-1'; }
    else if (/1\/2-1\/2|\bdraw\b|\bstalemate\b/i.test(text)) { evalPts = '1/2-1/2'; }
    else {
      var mateMatch = text.match(/[+-]?\s*M\s*(\d+)/i) || text.match(/#\s*([+-]?\d+)/);
      if (mateMatch) {
        var mateNum = parseInt(mateMatch[1], 10);
        evalPts = (mateMatch[0].indexOf('-') > -1) ? '-M' + mateNum : 'M' + mateNum;
      } else {
        var signedNums = text.match(/[+-]\d+\.?\d*/g);
        if (signedNums && signedNums.length > 0) {
          var v = parseFloat(signedNums[signedNums.length - 1]);
          evalPts = v.toFixed(2).replace(/\.?0+$/, '') || '0';
        } else {
          var decNums = text.match(/(?:^|\s)(\d+\.\d+)/g);
          if (decNums && decNums.length > 0) {
            var vd = parseFloat(decNums[decNums.length - 1]);
            evalPts = vd.toFixed(2).replace(/\.?0+$/, '') || '0';
          }
        }
      }
    }

    return { classification: classification, eval_points: evalPts };
  }

  function getGameId() {
    if (window.chesscom && window.chesscom.analysis && window.chesscom.analysis.gameId) {
      return window.chesscom.analysis.gameId;
    }
    var m = location.pathname.match(/\/(\d{6,})/);
    return m ? m[1] : Date.now().toString().slice(-6);
  }

  function downloadCSV(csv, filename) {
    var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
  }

  // ── Step 1: collect moves and SAN from DOM ─────────────────────────────────

  var moveNodes = Array.from(document.querySelectorAll('div.node.main-line-ply'));

  if (moveNodes.length === 0) {
    showStatus('No move nodes found. Is the Game Review loaded?', true);
    return;
  }

  var moves = [];
  for (var i = 0; i < moveNodes.length; i++) {
    var node = moveNodes[i];
    var dnParts = (node.getAttribute('data-node') || '').split('-');
    var ply = dnParts.length === 2 ? parseInt(dnParts[1], 10) + 1 : i + 1;
    var color = node.classList.contains('white-move') ? 'white' : 'black';

    var sanEl = node.querySelector('.node-highlight-content');
    var san = '';
    if (sanEl) {
      var figEl = sanEl.querySelector('[data-figurine]');
      var prefix = figEl ? figEl.getAttribute('data-figurine') : '';
      san = prefix + sanEl.textContent.trim();
    }

    moves.push({ ply: ply, color: color, move: san, classification: '', eval_points: '', points_difference: '' });
  }

  // ── Step 2: click each move, wait for feedback update, read class + eval ───

  var idx = 0;
  var lastText = '';
  var retries = 0;
  var MAX_RETRIES = 8;
  var POLL_MS = 150;

  showStatus('ChessLens: extracting ' + moves.length + ' moves (~' + Math.round(moves.length * 0.3) + 's)...');

  function readNext() {
    if (idx >= moves.length) {
      finish();
      return;
    }

    lastText = getFeedbackText();
    retries = 0;
    moveNodes[idx].click();
    pollFeedback();
  }

  function getFeedbackText() {
    var fb = document.querySelector('.move-feedback-component');
    return fb ? fb.textContent.trim() : '';
  }

  function pollFeedback() {
    setTimeout(function() {
      var text = getFeedbackText();

      if (text === lastText && retries < MAX_RETRIES && idx > 0) {
        retries++;
        pollFeedback();
        return;
      }

      var parsed = parseFeedback(text);
      moves[idx].classification = parsed.classification;
      moves[idx].eval_points = parsed.eval_points;

      if (idx > 0 && moves[idx].eval_points) {
        var curr = parseFloat(moves[idx].eval_points);
        var prev = parseFloat(moves[idx - 1].eval_points);
        if (!isNaN(curr) && !isNaN(prev)) {
          moves[idx].points_difference = (curr - prev).toFixed(2).replace(/\.?0+$/, '');
        }
      }

      idx++;
      readNext();
    }, POLL_MS);
  }

  // ── Step 3: build CSV and download ─────────────────────────────────────────

  function finish() {
    var header = 'ply,color,move,classification,eval_points,points_difference';
    var rows = moves.map(function(m) {
      return [m.ply, m.color, m.move, m.classification, m.eval_points, m.points_difference]
        .map(function(v) {
          var s = String(v || '');
          return s.indexOf(',') !== -1 ? '"' + s.replace(/"/g, '""') + '"' : s;
        })
        .join(',');
    });
    var csv = [header].concat(rows).join('\n');

    var gameId = getGameId();
    var filename = 'game_' + gameId + '.csv';
    downloadCSV(csv, filename);

    var classified = moves.filter(function(m) { return m.classification; }).length;
    var withEval = moves.filter(function(m) { return m.eval_points; }).length;
    showStatus('Extracted ' + moves.length + ' moves (' + classified + ' classified, ' + withEval + ' evals) -> ' + filename);
  }

  readNext();
})();
