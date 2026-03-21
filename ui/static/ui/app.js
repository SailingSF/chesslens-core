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
};
