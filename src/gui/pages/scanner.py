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

            # Scan button
            scan_btn = ui.button('Run Scan', on_click=lambda: asyncio.create_task(run_scan()))
            scan_btn.classes('bg-blue-600 hover:bg-blue-700')

            scan_status = ui.label('').classes('text-sm text-gray-400 ml-4')

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

            # Save config button
            async def save_config():
                bot_service.config['core_coins'] = core_coins_input.value.split()
                bot_service.config['max_dynamic_coins'] = int(max_dynamic.value)
                if bot_service.scanner:
                    bot_service.scanner.set_core_coins(bot_service.config['core_coins'])
                ui.notify('Scanner config saved!', type='positive')

            ui.button('Save Config', on_click=save_config).classes('bg-green-600 hover:bg-green-700 self-end')

    # ===== SCAN RESULTS =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Scan Results').classes('text-xl font-bold text-white mb-4')

        results_container = ui.column().classes('w-full')

        with results_container:
            ui.label('Click "Run Scan" to find opportunities').classes('text-gray-400 text-center py-8')

    # ===== TRADING ASSETS =====
    with ui.card().classes('w-full p-4'):
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('Current Trading Assets').classes('text-xl font-bold text-white')

            async def apply_scan_results():
                """Apply scan results as new trading assets"""
                symbols = bot_service.get_scanned_symbols()
                if symbols:
                    bot_service.config['assets'] = symbols
                    ui.notify(f'Trading assets updated: {", ".join(symbols)}', type='positive')
                    update_current_assets()
                else:
                    ui.notify('No scan results to apply. Run a scan first.', type='warning')

            ui.button('Apply Scan Results', on_click=apply_scan_results).classes('bg-purple-600 hover:bg-purple-700')

        current_assets_container = ui.row().classes('w-full gap-2 flex-wrap')

        def update_current_assets():
            current_assets_container.clear()
            with current_assets_container:
                for asset in bot_service.get_assets():
                    is_core = asset in bot_service.config.get('core_coins', [])
                    color = 'bg-blue-600' if is_core else 'bg-purple-600'
                    with ui.chip(asset).classes(f'{color} text-white'):
                        if is_core:
                            ui.tooltip('Core coin')

        update_current_assets()

    # ===== SCAN FUNCTION =====
    async def run_scan():
        scan_btn.disable()
        scan_status.text = 'Scanning...'

        try:
            results = await bot_service.scan_market(max_dynamic=int(max_dynamic.value))

            results_container.clear()
            with results_container:
                if results:
                    # Create table
                    columns = [
                        {'name': 'symbol', 'label': 'Symbol', 'field': 'symbol', 'sortable': True},
                        {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True},
                        {'name': 'score', 'label': 'Score', 'field': 'score', 'sortable': True},
                        {'name': 'signal', 'label': 'Signal', 'field': 'signal', 'sortable': True},
                        {'name': 'funding', 'label': 'Funding (APR)', 'field': 'funding_annualized', 'sortable': True},
                        {'name': 'volume', 'label': 'Volume 24h', 'field': 'volume_24h', 'sortable': True},
                        {'name': 'reasons', 'label': 'Reasons', 'field': 'reasons'},
                    ]

                    # Format data for table
                    rows = []
                    for r in results:
                        rows.append({
                            'symbol': r['symbol'],
                            'price': f"${r['price']:,.2f}" if r['price'] else 'N/A',
                            'score': f"{r['score']:.0f}",
                            'signal': r['signal'],
                            'funding_annualized': f"{r['funding_annualized']:.1f}%",
                            'volume_24h': f"${r['volume_24h']/1e6:.1f}M" if r['volume_24h'] else 'N/A',
                            'reasons': ', '.join(r['reasons'][:3]) if r['reasons'] else '-',
                        })

                    ui.table(
                        columns=columns,
                        rows=rows,
                        row_key='symbol'
                    ).classes('w-full')

                    # Summary cards
                    with ui.row().classes('w-full gap-4 mt-4'):
                        core_count = len([r for r in results if r['symbol'] in bot_service.config.get('core_coins', [])])
                        dynamic_count = len(results) - core_count

                        with ui.card().classes('p-4 bg-blue-900'):
                            ui.label(str(core_count)).classes('text-2xl font-bold text-blue-400')
                            ui.label('Core Coins').classes('text-sm text-gray-400')

                        with ui.card().classes('p-4 bg-purple-900'):
                            ui.label(str(dynamic_count)).classes('text-2xl font-bold text-purple-400')
                            ui.label('Dynamic Coins').classes('text-sm text-gray-400')

                        long_count = len([r for r in results if r['signal'] == 'LONG'])
                        with ui.card().classes('p-4 bg-green-900'):
                            ui.label(str(long_count)).classes('text-2xl font-bold text-green-400')
                            ui.label('Long Signals').classes('text-sm text-gray-400')

                        short_count = len([r for r in results if r['signal'] == 'SHORT'])
                        with ui.card().classes('p-4 bg-red-900'):
                            ui.label(str(short_count)).classes('text-2xl font-bold text-red-400')
                            ui.label('Short Signals').classes('text-sm text-gray-400')
                else:
                    ui.label('No opportunities found').classes('text-gray-400 text-center py-8')

            scan_status.text = f'Found {len(results)} opportunities'
            ui.notify(f'Scan complete: {len(results)} opportunities', type='positive')

        except Exception as e:
            scan_status.text = f'Error: {str(e)}'
            ui.notify(f'Scan failed: {str(e)}', type='negative')
        finally:
            scan_btn.enable()
