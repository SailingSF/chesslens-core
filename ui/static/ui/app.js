/*
 * ChessLens UI — shared utilities.
 *
 * API key is stored in localStorage under 'chesslens_api_key'.
 * All fetch calls in page-specific scripts should include the key header.
 */

window.ChessLens = {
    getApiKey() {
        return localStorage.getItem('chesslens_api_key') || '';
    },

    apiHeaders(extra) {
        const headers = {'Content-Type': 'application/json', ...extra};
        const key = this.getApiKey();
        if (key) headers['X-Anthropic-Key'] = key;
        return headers;
    },

    async populateEngineSelect(selectId) {
        const select = document.getElementById(selectId);
        if (!select) return;
        try {
            const resp = await fetch('/api/engines/');
            if (!resp.ok) throw new Error('engines request failed');
            const data = await resp.json();
            const engines = data.engines || [];
            if (engines.length === 0) {
                select.innerHTML = '<option value="">No engine discovered</option>';
                select.disabled = true;
                return;
            }
            select.innerHTML = '';
            for (const e of engines) {
                const opt = document.createElement('option');
                opt.value = e.id;
                opt.textContent = e.name;
                if (e.id === data.default_id) opt.selected = true;
                select.appendChild(opt);
            }
        } catch (err) {
            select.innerHTML = '<option value="">(engine list unavailable)</option>';
            select.disabled = true;
        }
    },
};
