
const chartOptions = {
    layout: {
        textColor: '#d1d4dc',
        background: { type: 'solid', color: '#131722' }
    },
    grid: {
        vertLines: { color: '#2a2e39' },
        horzLines: { color: '#2a2e39' }
    },
    crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
    },
    timeScale: {
        timeVisible: true,
        secondsVisible: false,
    },
};

let chart;
let candleSeries;
let tradesMap = new Map(); // time -> [trades]

document.addEventListener('DOMContentLoaded', async () => {
    // 1. Fetch Lists
    try {
        const res = await fetch('/api/datasets');
        const data = await res.json();

        const assetSelect = document.getElementById('asset-select');
        data.assets.forEach(a => {
            const opt = document.createElement('option');
            opt.value = a;
            opt.text = a;
            assetSelect.appendChild(opt);
        });

        const resultSelect = document.getElementById('result-select');
        data.results.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r;
            opt.text = r;
            resultSelect.appendChild(opt);
        });

        // Initialize empty chart
        const container = document.getElementById('chart-container');
        if (!container) {
            console.error("Chart container not found!");
            throw new Error("Chart container not found");
        }

        console.log("Initializing chart...");
        chart = LightweightCharts.createChart(container, chartOptions);
        candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350'
        });
        console.log("Chart initialized successfully");

        window.addEventListener('resize', () => {
            chart.applyOptions({ width: document.getElementById('chart-container').clientWidth, height: document.getElementById('chart-container').clientHeight });
        });

        // Subscribe to crosshair move for tooltip
        chart.subscribeCrosshairMove(param => {
            const tooltip = document.getElementById('tooltip');
            if (
                param.point === undefined ||
                !param.time ||
                param.point.x < 0 ||
                param.point.x > document.getElementById('chart-container').clientWidth ||
                param.point.y < 0 ||
                param.point.y > document.getElementById('chart-container').clientHeight
            ) {
                tooltip.style.display = 'none';
                return;
            }

            const dateStr = new Date(param.time * 1000).toLocaleString();

            // Check for trades at this time
            const trades = tradesMap.get(param.time);
            let tradeHtml = '';
            if (trades) {
                trades.forEach(t => {
                    const color = t.pnl >= 0 ? '#26a69a' : '#ef5350';
                    tradeHtml += `<div style="margin-top:4px; border-top:1px solid #444; padding-top:4px;">
                        <span style="color:${t.action === 'BUY' ? '#26a69a' : '#ef5350'}">${t.action}</span> 
                        Example PnL: <span style="color:${color}">${t.pnl.toFixed(2)}</span>
                    </div>`;

                    // Update sidebar details too (just showing the first one found for now)
                    document.getElementById('trade-details').style.display = 'block';
                    document.getElementById('td-action').innerText = t.action;
                    document.getElementById('td-action').style.color = t.action === 'BUY' ? 'green' : 'red';
                    document.getElementById('td-pnl').innerText = t.pnl.toFixed(2);
                    document.getElementById('td-pnl').style.color = color;
                    document.getElementById('td-entry').innerText = t.entry_price;
                    document.getElementById('td-exit').innerText = t.exit_price;
                    document.getElementById('td-time').innerText = dateStr;
                });
            }

            tooltip.style.display = 'block';
            tooltip.innerHTML = `<div>${dateStr}</div>${tradeHtml}`;
            tooltip.style.left = param.point.x + 'px';
            tooltip.style.top = param.point.y + 'px';
        });

        // Only attach button if initialization succeeded
        document.getElementById('load-btn').addEventListener('click', loadData);
        console.log("Setup complete. Chart and button ready.");

    } catch (e) {
        console.error("Init error", e);
        alert("Failed to initialize chart: " + e.message + ". Check console for details.");
        // Disable the button
        const btn = document.getElementById('load-btn');
        if (btn) {
            btn.disabled = true;
            btn.textContent = "Chart Failed to Load";
        }
    }
});

async function loadData() {
    const asset = document.getElementById('asset-select').value;
    const resultFile = document.getElementById('result-select').value;

    if (!asset || !resultFile) return alert("Select Asset and Result File");

    if (!chart || !candleSeries) {
        console.error("Chart or candleSeries not initialized", { chart, candleSeries });
        alert("Chart not properly initialized. Check console for errors.");
        return;
    }

    try {
        // Fetch Candles
        const cRes = await fetch(`/api/candles?asset=${asset}`);
        if (!cRes.ok) throw new Error(await cRes.text());
        const candles = await cRes.json();

        console.log("Loaded candles:", candles.length);
        candleSeries.setData(candles);

        // Fetch Trades
        const tRes = await fetch(`/api/trades?asset=${asset}&result_file=${resultFile}`);
        if (!tRes.ok) throw new Error(await tRes.text());
        const trades = await tRes.json();

        const markers = [];
        tradesMap.clear();

        trades.forEach(t => {
            // Find valid time in candles (sometimes trade time might be slightly off, but here we assume match)
            // Or we just use t.time.

            // Store in map
            if (!tradesMap.has(t.time)) tradesMap.set(t.time, []);
            tradesMap.get(t.time).push(t);

            // Marker for Entry
            markers.push({
                time: t.time,
                position: t.action === 'BUY' ? 'belowBar' : 'aboveBar',
                color: t.action === 'BUY' ? '#2196F3' : '#E91E63', // Blue for Buy, Pink for Sell entries
                shape: t.action === 'BUY' ? 'arrowUp' : 'arrowDown',
                text: t.action + (t.pnl < 0 ? ' (L)' : ' (W)'),
                size: 2
            });

            // Marker for Exit (approximate using hold_time assuming 5m candles)
            if (t.hold_time) {
                const exitTime = t.time + (t.hold_time * 5 * 60);
                // We verify if this time exists in chart?? Lightweight charts ignores markers with invalid time usually
                // But let's verify if within range.
                // For now, simpler: just entry.
            }
        });

        candleSeries.setMarkers(markers);
        chart.timeScale().fitContent();

    } catch (e) {
        alert("Error loading data: " + e.message);
    }
}
