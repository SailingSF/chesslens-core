/*
 * ChessLens UI — shared utilities.
 *
 * LLM settings are stored in localStorage (chosen on the home page; the
 * available models come from the server config in config/llm_models.py):
 *   chesslens_llm_provider  — "anthropic" (default) or "openai"
 *   chesslens_llm_model     — model slug; falls back to provider default
 *   chesslens_llm_effort    — reasoning effort: "low" | "medium" | "high"
 *   anthropic_api_key       — Anthropic API key override
 *   openai_api_key          — OpenAI API key override
 */

window.ChessLens = {

    getLLMProvider() {
        return localStorage.getItem('chesslens_llm_provider') || 'anthropic';
    },

    getLLMModel() {
        return localStorage.getItem('chesslens_llm_model') || '';
    },

    getLLMEffort() {
        return localStorage.getItem('chesslens_llm_effort') || '';
    },

    // Return the stored API key for the given (or currently selected) provider.
    getApiKey(provider) {
        provider = provider || this.getLLMProvider();
        return localStorage.getItem(
            provider === 'openai' ? 'openai_api_key' : 'anthropic_api_key'
        ) || '';
    },

    apiHeaders(extra) {
        const headers = { 'Content-Type': 'application/json', ...extra };
        const provider = this.getLLMProvider();
        const model    = this.getLLMModel();

        headers['X-LLM-Provider'] = provider;
        if (model) headers['X-LLM-Model'] = model;

        const effort = this.getLLMEffort();
        if (effort) headers['X-LLM-Reasoning'] = effort;

        const key = this.getApiKey(provider);
        if (key) {
            headers[provider === 'openai' ? 'X-OpenAI-Key' : 'X-Anthropic-Key'] = key;
        }

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
