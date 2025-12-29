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
                ui.label('Core Coins (excluded from scan)').classes('text-sm text-gray-300')
                core_coins_input = ui.input(
                    value=' '.join(bot_service.config.get('core_coins', ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX']))
                ).classes('w-64')
                ui.label('These are traded by main bot loop').classes('text-xs text-gray-500')

            with ui.column().classes('gap-2'):
                ui.label('Min Score').classes('text-sm text-gray-300')
                min_score_input = ui.number(
                    value=25,
                    min=10,
                    max=100
                ).classes('w-32')
                ui.label('Minimum opportunity score').classes('text-xs text-gray-500')

        with ui.row().classes('w-full gap-8 mt-4'):
            with ui.column().classes('gap-2'):
                ui.label('Allocation per Trade ($)').classes('text-sm text-gray-300')
                allocation_input = ui.number(
                    value=20.0,
                    min=5,
                    max=100
                ).classes('w-32')
                ui.label('USD per scanner trade').classes('text-xs text-gray-500')

            with ui.column().classes('gap-2'):
                ui.label('Max Trades').classes('text-sm text-gray-300')
                max_trades_input = ui.number(
                    value=3,
                    min=1,
                    max=5
                ).classes('w-32')
                ui.label('Max concurrent scanner positions').classes('text-xs text-gray-500')

        with ui.row().classes('w-full gap-4 mt-4'):
            # Scan button
            scan_btn = ui.button('Run Scan').classes('bg-blue-600 hover:bg-blue-700')

            # Save config button
            def save_config():
                bot_service.config['core_coins'] = core_coins_input.value.split()
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

    # ===== TRADING OPPORTUNITIES =====
    with ui.card().classes('w-full p-4 mb-6'):
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('Trading Opportunities').classes('text-xl font-bold text-white')
            ui.label('Excludes core coins and open positions').classes('text-sm text-gray-400')

        # Opportunities table
        opportunities_table = ui.table(
            columns=[
                {'name': 'symbol', 'label': 'Symbol', 'field': 'symbol', 'sortable': True, 'align': 'left'},
                {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True},
                {'name': 'score', 'label': 'Score', 'field': 'score', 'sortable': True},
                {'name': 'signal', 'label': 'Signal', 'field': 'signal', 'sortable': True},
                {'name': 'funding', 'label': 'Funding APR', 'field': 'funding'},
                {'name': 'reasons', 'label': 'Reasons', 'field': 'reasons'},
            ],
            rows=[],
            row_key='symbol'
        ).classes('w-full')

        with ui.row().classes('w-full gap-4 mt-4'):
            trade_btn = ui.button('Trade Opportunities').classes('bg-orange-600 hover:bg-orange-700')
            trade_status = ui.label('').classes('text-sm text-gray-400 ml-4')

    # ===== CURRENT STATUS =====
    with ui.card().classes('w-full p-4'):
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('Current Status').classes('text-xl font-bold text-white')

        with ui.row().classes('w-full gap-8'):
            with ui.column().classes('gap-1'):
                ui.label('Core Coins (Bot Loop)').classes('text-sm text-gray-400')
                current_assets_label = ui.label(', '.join(bot_service.get_assets())).classes('text-lg text-gray-300')

            with ui.column().classes('gap-1'):
                ui.label('Open Positions').classes('text-sm text-gray-400')
                open_positions_label = ui.label('Loading...').classes('text-lg text-gray-300')

    # ===== SCAN LOGIC =====
    async def do_scan():
        scan_btn.props('disabled=true')
        scan_status.set_text('Scanning...')

        try:
            # First do a full scan (includes core coins for display)
            results = await bot_service.scan_market(max_dynamic=10)

            if results:
                # Format rows for full results table
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

                core_count_label.set_text(str(core_count))
                dynamic_count_label.set_text(str(dynamic_count))
                long_count_label.set_text(str(long_count))
                short_count_label.set_text(str(short_count))

                # Now get trading opportunities (excludes core + open positions)
                opportunities = bot_service.get_trading_opportunities(
                    min_score=float(min_score_input.value)
                )

                if opportunities:
                    opp_rows = []
                    for r in opportunities:
                        opp_rows.append({
                            'symbol': r['symbol'],
                            'price': f"${r['price']:,.2f}" if r['price'] else 'N/A',
                            'score': f"{r['score']:.0f}",
                            'signal': r['signal'],
                            'funding': f"{r['funding_annualized']:.1f}%",
                            'reasons': ', '.join(r['reasons'][:2]) if r['reasons'] else '-',
                        })
                    opportunities_table.rows = opp_rows
                    opportunities_table.update()
                    trade_status.set_text(f'{len(opportunities)} tradeable opportunities')
                else:
                    opportunities_table.rows = []
                    opportunities_table.update()
                    trade_status.set_text('No tradeable opportunities (all filtered)')

                scan_status.set_text(f'Found {len(results)} total, {len(opportunities)} tradeable')

                # Update open positions display
                state = bot_service.get_state()
                if state.positions:
                    pos_symbols = [p.get('symbol', '?') for p in state.positions]
                    open_positions_label.set_text(', '.join(pos_symbols))
                else:
                    open_positions_label.set_text('None')

            else:
                results_table.rows = []
                results_table.update()
                opportunities_table.rows = []
                opportunities_table.update()
                scan_status.set_text('No opportunities found')

        except Exception as e:
            scan_status.set_text(f'Error: {str(e)[:50]}')
        finally:
            scan_btn.props('disabled=false')

    async def do_trade():
        trade_btn.props('disabled=true')
        trade_status.set_text('Executing trades...')

        try:
            opportunities = bot_service.get_trading_opportunities(
                min_score=float(min_score_input.value)
            )

            if not opportunities:
                trade_status.set_text('No opportunities to trade')
                ui.notify('No trading opportunities available', type='warning')
                return

            results = await bot_service.execute_scanner_trades(
                opportunities=opportunities,
                max_trades=int(max_trades_input.value),
                allocation_per_trade=float(allocation_input.value)
            )

            executed = len([r for r in results if r.get('status') == 'executed'])
            failed = len([r for r in results if r.get('status') in ('failed', 'error')])

            if executed > 0:
                trade_status.set_text(f'Executed {executed} trades')
                ui.notify(f'Successfully executed {executed} scanner trades!', type='positive')

                # Refresh scan to update opportunities
                await do_scan()
            else:
                trade_status.set_text(f'No trades executed ({failed} failed)')
                ui.notify(f'No trades executed. {failed} failed.', type='warning')

        except Exception as e:
            trade_status.set_text(f'Error: {str(e)[:50]}')
            ui.notify(f'Trade execution error: {str(e)}', type='negative')
        finally:
            trade_btn.props('disabled=false')

    # Connect buttons using on_click (sync wrapper)
    scan_btn.on_click(lambda: asyncio.create_task(do_scan()))
    trade_btn.on_click(lambda: asyncio.create_task(do_trade()))
