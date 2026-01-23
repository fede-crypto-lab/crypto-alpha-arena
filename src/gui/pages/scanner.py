"""
Scanner Page - Universal market opportunity scanner with multi-source data
"""

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

        # Configuration Row 1
        with ui.row().classes('w-full gap-8'):
            with ui.column().classes('gap-2'):
                ui.label('Scanner Mode').classes('text-sm text-gray-300')
                scanner_mode = ui.toggle(
                    {True: 'Market-Wide', False: 'Exchange Only'},
                    value=True
                ).classes('w-48')
                ui.label('Market-Wide uses CoinGecko + exchange check').classes('text-xs text-gray-500')

            with ui.column().classes('gap-2'):
                ui.label('Core Coins (excluded from trading)').classes('text-sm text-gray-300')
                core_coins_input = ui.input(
                    value=' '.join(bot_service.config.get('core_coins', ['BTC', 'ETH', 'SOL', 'DOGE', 'AVAX']))
                ).classes('w-64')
                ui.label('Bot trades these in main loop').classes('text-xs text-gray-500')

            with ui.column().classes('gap-2'):
                ui.label('Min Score').classes('text-sm text-gray-300')
                min_score_input = ui.number(
                    value=25,
                    min=10,
                    max=100
                ).classes('w-32')
                ui.label('Minimum opportunity score').classes('text-xs text-gray-500')

        # Configuration Row 2
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

            with ui.column().classes('gap-2'):
                ui.label('Top N Coins').classes('text-sm text-gray-300')
                top_n_input = ui.number(
                    value=50,
                    min=20,
                    max=100
                ).classes('w-32')
                ui.label('Coins to scan from market').classes('text-xs text-gray-500')

        with ui.row().classes('w-full gap-4 mt-4'):
            # Scan button
            scan_btn = ui.button('Run Scan').classes('bg-blue-600 hover:bg-blue-700')

            # Save config button
            def save_config():
                bot_service.config['core_coins'] = core_coins_input.value.split()
                bot_service.use_universal_scanner = scanner_mode.value
                if bot_service.scanner:
                    bot_service.scanner.set_core_coins(bot_service.config['core_coins'])
                if bot_service.universal_scanner:
                    bot_service.universal_scanner.set_core_coins(bot_service.config['core_coins'])
                ui.notify('Scanner config saved!', type='positive')

            ui.button('Save Config', on_click=save_config).classes('bg-green-600 hover:bg-green-700')

    # ===== SCAN RESULTS TABLE =====
    with ui.card().classes('w-full p-4 mb-6'):
        ui.label('Scan Results').classes('text-xl font-bold text-white mb-4')

        # Results table with new columns for universal scanner
        results_table = ui.table(
            columns=[
                {'name': 'rank', 'label': '#', 'field': 'rank', 'sortable': True, 'align': 'center'},
                {'name': 'symbol', 'label': 'Symbol', 'field': 'symbol', 'sortable': True, 'align': 'left'},
                {'name': 'name', 'label': 'Name', 'field': 'name', 'align': 'left'},
                {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True},
                {'name': 'change_1h', 'label': '1h %', 'field': 'change_1h', 'sortable': True},
                {'name': 'supertrend', 'label': 'Trend 4h', 'field': 'supertrend', 'sortable': True},
                {'name': 'score', 'label': 'Score', 'field': 'score', 'sortable': True},
                {'name': 'signal', 'label': 'Signal', 'field': 'signal', 'sortable': True},
                {'name': 'tradeable', 'label': 'Tradeable', 'field': 'tradeable', 'sortable': True},
                {'name': 'funding', 'label': 'Funding APR', 'field': 'funding'},
                {'name': 'reasons', 'label': 'Reasons', 'field': 'reasons'},
            ],
            rows=[],
            row_key='symbol'
        ).classes('w-full')

    # ===== SUMMARY STATS =====
    with ui.card().classes('w-full p-4 mb-6'):
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('p-4 bg-blue-900 flex-1'):
                total_scanned_label = ui.label('0').classes('text-2xl font-bold text-blue-400')
                ui.label('Total Scanned').classes('text-sm text-gray-400')

            with ui.card().classes('p-4 bg-green-900 flex-1'):
                tradeable_count_label = ui.label('0').classes('text-2xl font-bold text-green-400')
                ui.label('Tradeable').classes('text-sm text-gray-400')

            with ui.card().classes('p-4 bg-purple-900 flex-1'):
                long_count_label = ui.label('0').classes('text-2xl font-bold text-purple-400')
                ui.label('Long Signals').classes('text-sm text-gray-400')

            with ui.card().classes('p-4 bg-red-900 flex-1'):
                short_count_label = ui.label('0').classes('text-2xl font-bold text-red-400')
                ui.label('Short Signals').classes('text-sm text-gray-400')

    # ===== TRADING OPPORTUNITIES =====
    with ui.card().classes('w-full p-4 mb-6'):
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('Trading Opportunities').classes('text-xl font-bold text-white')
            ui.label('Excludes core coins and open positions').classes('text-sm text-gray-400')

        # Opportunities table (only shows trend-aligned opportunities)
        opportunities_table = ui.table(
            columns=[
                {'name': 'symbol', 'label': 'Symbol', 'field': 'symbol', 'sortable': True, 'align': 'left'},
                {'name': 'name', 'label': 'Name', 'field': 'name', 'align': 'left'},
                {'name': 'price', 'label': 'Price', 'field': 'price', 'sortable': True},
                {'name': 'change_1h', 'label': '1h %', 'field': 'change_1h', 'sortable': True},
                {'name': 'supertrend', 'label': 'Trend 4h', 'field': 'supertrend', 'sortable': True},
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

            with ui.column().classes('gap-1'):
                ui.label('Scanner Mode').classes('text-sm text-gray-400')
                scanner_mode_label = ui.label('Market-Wide').classes('text-lg text-gray-300')

    # ===== SCAN LOGIC =====
    async def do_scan():
        scan_btn.props('disabled=true')
        scan_status.set_text('Scanning...')

        try:
            # Update scanner mode
            bot_service.use_universal_scanner = scanner_mode.value
            scanner_mode_label.set_text('Market-Wide' if scanner_mode.value else 'Exchange Only')

            # Run scan
            results = await bot_service.scan_market(max_dynamic=int(top_n_input.value))

            if results:
                # Format rows for results table
                rows = []
                for i, r in enumerate(results):
                    # Handle both universal and old scanner formats
                    is_tradeable = r.get('exchange_available', True)  # Default to True for old scanner
                    name = r.get('name', '')[:15]  # Truncate long names
                    change_1h = r.get('price_change_1h', 0)
                    change_24h = r.get('price_change_24h', 0)

                    # Format supertrend with emoji
                    supertrend = r.get('supertrend', '')
                    if supertrend == 'LONG':
                        supertrend_display = '🟢 LONG'
                    elif supertrend == 'SHORT':
                        supertrend_display = '🔴 SHORT'
                    else:
                        supertrend_display = '-'

                    rows.append({
                        'rank': r.get('market_cap_rank', i + 1),
                        'symbol': r['symbol'],
                        'name': name,
                        'price': f"${r['price']:,.4f}" if r['price'] < 1 else f"${r['price']:,.2f}" if r.get('price') else 'N/A',
                        'change_1h': f"{change_1h:+.1f}%" if change_1h else '-',
                        'supertrend': supertrend_display,
                        'score': f"{r['score']:.0f}",
                        'signal': r['signal'],
                        'tradeable': '✓' if is_tradeable else '✗',
                        'funding': f"{r.get('funding_annualized', 0):.1f}%",
                        'reasons': ', '.join(r.get('reasons', [])[:2]) if r.get('reasons') else '-',
                    })
                results_table.rows = rows
                results_table.update()

                # Update stats
                total_scanned = len(results)
                tradeable = len([r for r in results if r.get('exchange_available', True)])
                long_count = len([r for r in results if r['signal'] == 'LONG'])
                short_count = len([r for r in results if r['signal'] == 'SHORT'])

                total_scanned_label.set_text(str(total_scanned))
                tradeable_count_label.set_text(str(tradeable))
                long_count_label.set_text(str(long_count))
                short_count_label.set_text(str(short_count))

                # Get trading opportunities (excludes core + open positions)
                opportunities = bot_service.get_trading_opportunities(
                    min_score=float(min_score_input.value)
                )

                if opportunities:
                    opp_rows = []
                    for r in opportunities:
                        name = r.get('name', '')[:15]
                        change_1h = r.get('price_change_1h', 0)

                        # Format supertrend with emoji
                        supertrend = r.get('supertrend', '')
                        if supertrend == 'LONG':
                            supertrend_display = '🟢 LONG'
                        elif supertrend == 'SHORT':
                            supertrend_display = '🔴 SHORT'
                        else:
                            supertrend_display = '-'

                        opp_rows.append({
                            'symbol': r['symbol'],
                            'name': name,
                            'price': f"${r['price']:,.4f}" if r['price'] < 1 else f"${r['price']:,.2f}" if r.get('price') else 'N/A',
                            'change_1h': f"{change_1h:+.1f}%" if change_1h else '-',
                            'supertrend': supertrend_display,
                            'score': f"{r['score']:.0f}",
                            'signal': r['signal'],
                            'funding': f"{r.get('funding_annualized', 0):.1f}%",
                            'reasons': ', '.join(r.get('reasons', [])[:2]) if r.get('reasons') else '-',
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

    # Connect buttons - use async handlers directly (NiceGUI handles this)
    scan_btn.on_click(do_scan)
    trade_btn.on_click(do_trade)
