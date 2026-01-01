"""
Main GUI Application - Single Page App with internal navigation
"""

import asyncio
import logging
from nicegui import ui
from src.gui.components.header import create_header
from src.gui.services.bot_service import BotService
from src.gui.services.state_manager import StateManager
from src.backend.config_loader import CONFIG

# Import pages
from src.gui.pages import dashboard, positions, history, market, reasoning, recommendations, scanner
# settings page removed for security (API keys visible from outside)

logger = logging.getLogger(__name__)

# Global services
bot_service = BotService()
state_manager = StateManager()

# Connect services
bot_service.state_manager = state_manager

# Track if autostart has been triggered
_autostart_done = False


def create_app():
    """Initialize and configure the NiceGUI application as single page"""

    # Add Material Icons font
    ui.add_head_html('<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">')
    
    # Add global styles
    ui.add_head_html('''
        <style>
            * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .positive { color: #10b981 !important; }
            .negative { color: #ef4444 !important; }

            .q-btn {
                text-transform: none !important;
            }

            /* Custom scrollbar for dark theme */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }

            ::-webkit-scrollbar-track {
                background: #1f2937;
            }

            ::-webkit-scrollbar-thumb {
                background: #4b5563;
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: #6b7280;
            }

            /* Responsive layout */
            @media (max-width: 1200px) {
                .metric-card {
                    padding: 16px;
                }
                .metric-card .text-4xl {
                    font-size: 1.5rem;
                }
            }

            @media (max-width: 900px) {
                .sidebar-nav {
                    width: 200px !important;
                }
            }

            /* Ensure content fills available space */
            .content-area {
                min-width: 0;
                width: 100%;
            }
        </style>
    ''')

    # Main layout
    with ui.column().classes('w-full h-screen'):
        create_header(state_manager)

        with ui.row().classes('w-full flex-grow flex-nowrap'):
            # Sidebar with navigation (flex-shrink-0 prevents it from shrinking)
            with ui.column().classes('w-64 bg-gray-800 p-4 gap-2 h-full flex-shrink-0'):
                # Navigation buttons
                menu_items = [
                    ('Dashboard', '📊 Dashboard', 'Main dashboard with metrics'),
                    ('Scanner', '🔍 Scanner', 'Find trading opportunities'),
                    ('Recommendations', '🤖 AI Recommendations', 'Review and approve AI trade proposals'),
                    ('Positions', '💼 Positions', 'Active trading positions'),
                    ('History', '📜 History', 'Trade history and logs'),
                    ('Market', '📈 Market', 'Market data and indicators'),
                    ('Reasoning', '🧠 AI Reasoning', 'LLM decision logic'),
                    # Settings removed for security (API keys visible from outside)
                ]

                # Create navigation buttons
                for page_id, label, tooltip in menu_items:
                    btn = ui.button(label, on_click=lambda p=page_id: navigate(p))
                    btn.classes('w-full justify-start text-left')
                    btn.props('flat')
                    with btn:
                        ui.tooltip(tooltip)

                ui.separator().classes('my-4')

                # Version info
                with ui.column().classes('mt-auto gap-1'):
                    ui.label('Version 1.0.0').classes('text-xs text-gray-500 text-center')
                    ui.label('Powered by NiceGUI').classes('text-xs text-gray-600 text-center')

            # Main content area (will be updated by navigation)
            global content_container
            content_container = ui.column().classes('flex-grow p-6 overflow-auto content-area')

    # Load default page
    with content_container:
        dashboard.create_dashboard(bot_service, state_manager)

    # Auto-start bot if AUTOSTART_BOT=true (using timer to run after UI is ready)
    async def _try_autostart(attempt: int = 1, max_attempts: int = 5):
        global _autostart_done
        if _autostart_done or bot_service.is_running():
            return

        if not CONFIG.get("autostart_bot"):
            return

        logger.info(f"[AUTOSTART] Attempt {attempt}/{max_attempts} - starting bot...")
        try:
            await bot_service.start()
            _autostart_done = True
            logger.info("[AUTOSTART] Bot started successfully!")
        except Exception as e:
            error_str = str(e)
            # Retry on rate limit (429) or connection errors
            if attempt < max_attempts and ("429" in error_str or "Connection" in error_str or "timeout" in error_str.lower()):
                wait_time = 5 * attempt  # 5s, 10s, 15s, 20s, 25s
                logger.warning(f"[AUTOSTART] Rate limited, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                await _try_autostart(attempt + 1, max_attempts)
            else:
                logger.error(f"[AUTOSTART] Failed to start bot after {attempt} attempts: {e}")

    # Trigger autostart after 3 seconds (ensures UI is fully loaded)
    if CONFIG.get("autostart_bot"):
        ui.timer(3.0, _try_autostart, once=True)


def navigate(page: str):
    """Navigate to different page by clearing and recreating content"""
    global content_container

    content_container.clear()

    with content_container:
        if page == 'Dashboard':
            dashboard.create_dashboard(bot_service, state_manager)
        elif page == 'Scanner':
            scanner.create_scanner(bot_service, state_manager)
        elif page == 'Recommendations':
            recommendations.create_recommendations(bot_service, state_manager)
        elif page == 'Positions':
            positions.create_positions(bot_service, state_manager)
        elif page == 'History':
            history.create_history(bot_service, state_manager)
        elif page == 'Market':
            market.create_market(bot_service, state_manager)
        elif page == 'Reasoning':
            reasoning.create_reasoning(bot_service, state_manager)
        # Settings page removed for security
