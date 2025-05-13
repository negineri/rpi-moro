// 即時関数を使用して早期初期化
(function () {
    // socket.ioの接続をすぐに開始
    const socket = io.connect(window.location.origin, {
        reconnectionDelayMax: 1000,
        transports: ['websocket'], // websocketを優先して使用
        forceNew: true
    });

    // DOMContentLoadedを待たずにsocketの初期化
    let streamElement;
    let fpsCounter;
    let cameraInfo;
    let receivedFirstFrame = false;
    let loadingOverlay;

    // フレームカウンター変数
    let frameCount = 0;
    let lastTime = new Date().getTime();

    // フレーム受信ハンドラを即時設定
    socket.on('frame', function (data) {
        // DOM要素がまだ取得されていない場合は取得
        if (!streamElement) {
            streamElement = document.getElementById('stream');
        }

        if (streamElement && data.frame) {
            streamElement.src = 'data:image/jpeg;base64,' + data.frame;

            // 最初のフレームを受信したら読み込みオーバーレイを非表示
            if (!receivedFirstFrame) {
                receivedFirstFrame = true;
                if (loadingOverlay) {
                    loadingOverlay.style.opacity = '0';
                    setTimeout(() => {
                        loadingOverlay.style.display = 'none';
                    }, 500);
                }
            }

            // FPSカウンターの更新
            frameCount++;
            const now = new Date().getTime();
            if (fpsCounter && now - lastTime >= 1000) {
                const fps = frameCount / ((now - lastTime) / 1000);
                fpsCounter.innerText = fps.toFixed(1);
                frameCount = 0;
                lastTime = now;
            }
        }
    });

    // カメラ情報受信ハンドラも設定
    socket.on('camera_info', function (info) {
        if (!cameraInfo) {
            cameraInfo = document.getElementById('camera-info');
        }

        if (cameraInfo && info) {
            let infoHTML = '<h3>カメラ情報</h3>';
            infoHTML += '<ul>';
            for (const [key, value] of Object.entries(info)) {
                infoHTML += `<li>${key}: ${value}</li>`;
            }
            infoHTML += '</ul>';
            cameraInfo.innerHTML = infoHTML;
        }
    });

    // DOMContentLoadedでDOMへの参照を取得
    document.addEventListener('DOMContentLoaded', function () {
        streamElement = document.getElementById('stream');
        fpsCounter = document.getElementById('fps-counter');
        cameraInfo = document.getElementById('camera-info');
        loadingOverlay = document.getElementById('loading-overlay');

        // 接続状態を確認し、必要なら明示的に再接続
        if (!socket.connected) {
            socket.connect();
        }
    });
})();
