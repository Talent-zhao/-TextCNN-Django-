/**
 * 路径输入增强：在「浏览」按钮后从本机选择文件，将相对路径填入（默认 model/文件名）。
 * 浏览器出于安全不会暴露完整本机路径；请把文件放到项目对应目录或复制文件名。
 */
(function () {
    function normalizePrefix(p) {
        var s = (p || 'model/').replace(/\\/g, '/').trim();
        if (!s) return 'model/';
        if (!s.endsWith('/')) s += '/';
        return s;
    }

    function bindOne(wrap) {
        var input = wrap.querySelector('input[type="text"], input:not([type])');
        var btn = wrap.querySelector('.btn-path-file');
        if (!input || !btn) return;

        var prefix = normalizePrefix(wrap.getAttribute('data-path-prefix'));
        var accept = wrap.getAttribute('data-accept') || '';

        var hidden = document.createElement('input');
        hidden.type = 'file';
        hidden.style.position = 'absolute';
        hidden.style.left = '-9999px';
        hidden.setAttribute('aria-hidden', 'true');
        if (accept) hidden.setAttribute('accept', accept);
        wrap.appendChild(hidden);

        btn.addEventListener('click', function (e) {
            e.preventDefault();
            hidden.value = '';
            hidden.click();
        });

        hidden.addEventListener('change', function () {
            if (!hidden.files || !hidden.files.length) return;
            var name = hidden.files[0].name;
            if (!name) return;
            // 由于浏览器无法给出用户选中文件的目录信息，
            // 若当前输入已经带有目录前缀（如 model/svm/），则优先沿用该目录。
            var cur = (input.value || '').trim();
            if (cur && cur.indexOf('/') >= 0) {
                var idx = cur.lastIndexOf('/') + 1;
                var dir = cur.slice(0, idx);
                if (dir && dir.startsWith('model/')) {
                    prefix = normalizePrefix(dir);
                }
            }
            input.value = prefix + name;
            try {
                input.dispatchEvent(new Event('change', { bubbles: true }));
            } catch (err) {}
        });
    }

    function init() {
        var wraps = document.querySelectorAll('.path-input-wrap');
        for (var i = 0; i < wraps.length; i++) bindOne(wraps[i]);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
