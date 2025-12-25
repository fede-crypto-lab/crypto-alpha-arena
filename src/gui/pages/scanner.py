"""
Scanner Page - Market opportunity scanner with dynamic coin selection
"""

import asyncio
from nicegui import ui
from src.gui.services.bot_service import BotService
from src.gui.services.state_manager import StateManager


def create_scanner(bot_service: BotService, state_manager: StateManager):
    """Create market scanner page"""

    ui.label('Market Scanner').classes('text-3xl font-bold mb-4 text-white')

    # ===== SCANNER CONTROLS =====
    with ui.card().classes('w-full p-4 mb-6'):
        with ui.row().classes('w-full items-center gap-4'):
            ui.label('Scan for Trading Opportunities').classes('text-xl font-bold text-white')

            scan_status = ui.label('').classes('text-sm text-gray-400 ml-auto')

        ui.separator().classes('my-4')

        # Configuration
        with ui.row().classes('w-full gap-8'):
            with ui.column().classes('gap-2'):
                ui.label('Core Coins (always traded)').classes('text-sm text-gray-300')
                core_coins_input = ui.input(
                    value=' '.join(bot_service.config.get('core_coins', ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP']))
                ).classes('w-64')
                ui.label('Space-separated symbols').classes('text-xs text-gray-500')

            with ui.column().classes('gap-2'):
                ui.label('Max Dynamic Coins').classes('text-sm text-gray-300')
                max_dynamic = ui.number(
                    value=bot_service.config.get('max_dynamic_coins', 3),
                    min=0,
                    max=10
                ).classes('w-32')
                ui.label('Additional coins from scan').classes('text-xs text-gray-500')

        with ui.row().classes('w-full gap-4 mt-4'):
            # Scan button
            scan_btn = ui.button('Run Scan').classes('bg-blue-600 hover:bg-blue-700')

            # Save config button
            def save_config():
                bot_service.config['core_coins'] = core_coins_input.value.split()
                bot_service.config['max_dynamic_coins'] = int(max_dynamic.value)
                if bot_service.scanner:
                    bot_service.scanner.set_core_coins(bot_service.config['core_coins'])
                ui.notify('Scanner config saved!', type='positive')

            ui.button('Save Config', on_click=save_config).classes('bg-green-600 hover:bg-green-700')

    # ===== SCAN RESULTS TABLE =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Scan Results').classes('text-xl font-bold text-white mb-4')

        # Results table (initially empty)
        results_table = ui.table(
            columns=[
                {'name': 'symbol', 'label': 'Symbol', 'field': 'symbol', 'sortable': True, 'align': 'left'},
                {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True},
                {'name': 'score', 'label': 'Score', 'field': 'score', 'sortable': True},
                {'name': 'signal', 'label': 'Signal', 'field': 'signal', 'sortable': True},
                {'name': 'funding', 'label': 'Funding APR', 'field': 'funding'},
                {'name': 'volume', 'label': 'Volume 24h', 'field': 'volume'},
                {'name': 'reasons', 'label': 'Reasons', 'field': 'reasons'},
            ],
            rows=[],
            row_key='symbol'
        ).classes('w-full')

    # ===== SUMMARY STATS =====
    with ui.card().classes('w-full p-4 mb-6'):
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('p-4 bg-blue-900 flex-1'):
                core_count_label = ui.label('0').classes('text-2xl font-bold text-blue-400')
                ui.label('Core Coins').classes('text-sm text-gray-400')

            with ui.card().classes('p-4 bg-purple-900 flex-1'):
                dynamic_count_label = ui.label('0').classes('text-2xl font-bold text-purple-400')
                ui.label('Dynamic Coins').classes('text-sm text-gray-400')

            with ui.card().classes('p-4 bg-green-900 flex-1'):
                long_count_label = ui.label('0').classes('text-2xl font-bold text-green-400')
                ui.label('Long Signals').classes('text-sm text-gray-400')

            with ui.card().classes('p-4 bg-red-900 flex-1'):
                short_count_label = ui.label('0').classes('text-2xl font-bold text-red-400')
                ui.label('Short Signals').classes('text-sm text-gray-400')

    # ===== CURRENT TRADING ASSETS =====
    with ui.card().classes('w-full p-4'):
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('Current Trading Assets').classes('text-xl font-bold text-white')

            apply_btn = ui.button('Apply Scan Results').classes('bg-purple-600 hover:bg-purple-700')

        current_assets_label = ui.label(', '.join(bot_service.get_assets())).classes('text-lg text-gray-300')

    # ===== SCAN LOGIC =====
    async def do_scan():
        scan_btn.disable()
        scan_status.text = 'Scanning...'

        try:
            results = await bot_service.scan_market(max_dynamic=int(max_dynamic.value))

            if results:
                # Format rows for table
                rows = []
                for r in results:
                    rows.append({
                        'symbol': r['symbol'],
                        'price': f"${r['price']:,.2f}" if r['price'] else 'N/A',
                        'score': f"{r['score']:.0f}",
                        'signal': r['signal'],
                        'funding': f"{r['funding_annualized']:.1f}%",
                        'volume': f"${r['volume_24h']/1e6:.1f}M" if r['volume_24h'] else 'N/A',
                        'reasons': ', '.join(r['reasons'][:2]) if r['reasons'] else '-',
                    })
                results_table.rows = rows
                results_table.update()

                # Update stats
                core_coins = bot_service.config.get('core_coins', [])
                core_count = len([r for r in results if r['symbol'] in core_coins])
                dynamic_count = len(results) - core_count
                long_count = len([r for r in results if r['signal'] == 'LONG'])
                short_count = len([r for r in results if r['signal'] == 'SHORT'])

                core_count_label.text = str(core_count)
                dynamic_count_label.text = str(dynamic_count)
                long_count_label.text = str(long_count)
                short_count_label.text = str(short_count)

                scan_status.text = f'Found {len(results)} opportunities'
                ui.notify(f'Scan complete: {len(results)} opportunities', type='positive')
            else:
                results_table.rows = []
                results_table.update()
                scan_status.text = 'No opportunities found'
                ui.notify('No opportunities found', type='warning')

        except Exception as e:
            scan_status.text = f'Error: {str(e)[:50]}'
            ui.notify(f'Scan failed: {str(e)}', type='negative')
        finally:
            scan_btn.enable()

    def apply_results():
        symbols = bot_service.get_scanned_symbols()
        if symbols:
            bot_service.config['assets'] = symbols
            current_assets_label.text = ', '.join(symbols)
            ui.notify(f'Trading assets updated: {len(symbols)} coins', type='positive')
        else:
            ui.notify('No scan results. Run a scan first.', type='warning')

    # Connect buttons
    scan_btn.on('click', lambda: asyncio.create_task(do_scan()))
    apply_btn.on('click', apply_results)
