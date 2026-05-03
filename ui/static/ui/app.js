/*
 * ChessLens UI — shared utilities.
 *
 * LLM settings are stored in localStorage:
 *   chesslens_llm_provider  — "anthropic" (default) or "openai"
 *   chesslens_llm_model     — model name; falls back to provider default
 *   anthropic_api_key       — Anthropic API key override
 *   openai_api_key          — OpenAI API key override
 */

window.ChessLens = {

    // Model lists shown in the provider selector on the home page.
    MODELS: {
        anthropic: [
            { value: 'claude-sonnet-4-6',        label: 'Claude Sonnet 4.6' },
            { value: 'claude-opus-4-7',           label: 'Claude Opus 4.7' },
            { value: 'claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5' },
        ],
        openai: [
            { value: 'gpt-5.5',     label: 'GPT-5.5 (reasoning)' },
            { value: 'gpt-4.1',     label: 'GPT-4.1' },
            { value: 'gpt-4.1-mini',label: 'GPT-4.1 Mini' },
            { value: 'gpt-4o',      label: 'GPT-4o' },
            { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
            { value: 'o3',          label: 'o3 (reasoning)' },
            { value: 'o4-mini',     label: 'o4-mini (reasoning)' },
        ],
    },

    DEFAULT_MODELS: { anthropic: 'claude-sonnet-4-6', openai: 'gpt-5.5' },

    getLLMProvider() {
        return localStorage.getItem('chesslens_llm_provider') || 'anthropic';
    },

    getLLMModel() {
        return localStorage.getItem('chesslens_llm_model') || '';
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
