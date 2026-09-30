"""Consistent, isolated homepage captures with bounded waits."""
import asyncio
import os


async def capture_sites(dataset, args):
    from playwright.async_api import async_playwright

    rows = dataset.capture_queue(args.retry_failed, args.limit)
    if not rows:
        print('No pending captures. Use --retry-failed to retry failures.')
        return
    jobs = asyncio.Queue()
    for row in rows:
        jobs.put_nowait(row)
    settings = {
        'viewport': {'width': args.width, 'height': args.height},
        'device_scale_factor': 1,
        'color_scheme': 'light',
        'locale': 'en-US',
        'timezone_id': 'UTC',
        'cookie_policy': 'leave banners as displayed; manually skip obscured pages',
        'settle_seconds': args.settle,
        'navigation_timeout_seconds': args.timeout,
        'full_page': False,
        'animations': 'disabled',
        'proxy_used': bool(args.proxy),
    }
    async with async_playwright() as playwright:
        launch = {'headless': not args.headed}
        if args.proxy:
            launch['proxy'] = {'server': args.proxy}
        browser = await playwright.chromium.launch(**launch)
        settings['browser_version'] = browser.version
        done = 0

        async def worker():
            nonlocal done
            while not jobs.empty():
                row = jobs.get_nowait()
                site_id = row['id']
                context = None
                temporary = dataset.image(site_id).with_suffix('.part.png')
                try:
                    # A fresh context per domain avoids inherited cookies and sessions.
                    context = await browser.new_context(
                        viewport=settings['viewport'], device_scale_factor=1,
                        color_scheme='light', locale='en-US', timezone_id='UTC',
                        accept_downloads=False,
                    )
                    page = await context.new_page()
                    page.set_default_timeout(args.timeout * 1000)
                    page.on('dialog', lambda dialog: asyncio.create_task(dialog.dismiss()))
                    response = await page.goto('https://' + row['domain'], wait_until='domcontentloaded')
                    if response is None:
                        raise RuntimeError('No homepage response')
                    status = response.status
                    if status >= 400:
                        raise RuntimeError(f'HTTP {status}')
                    content_type = response.headers.get('content-type', '').lower()
                    if 'html' not in content_type:
                        raise RuntimeError('Homepage response is not HTML')
                    await page.wait_for_timeout(args.settle * 1000)
                    # Font loading gets a bounded extra wait; active connections need not become idle.
                    await page.evaluate('''async () => {
                        await Promise.race([document.fonts.ready,
                            new Promise(resolve => setTimeout(resolve, 2000))]);
                        window.scrollTo(0, 0);
                    }''')
                    title = await page.title()
                    final_url = page.url
                    await page.screenshot(path=str(temporary), full_page=False,
                                          animations='disabled', caret='hide')
                    os.replace(temporary, dataset.image(site_id))
                    dataset.captured(site_id, final_url=final_url, title=title,
                                     http_status=status, settings=settings)
                    result = 'saved'
                except Exception as exc:
                    dataset.failed(site_id, str(exc))
                    result = f'failed: {str(exc).splitlines()[0][:160]}'
                finally:
                    temporary.unlink(missing_ok=True)
                    if context is not None:
                        await context.close()
                    jobs.task_done()
                done += 1
                print(f'[{done}/{len(rows)}] {row["domain"]}: {result}', flush=True)
                if not jobs.empty():
                    await asyncio.sleep(args.delay)

        try:
            await asyncio.gather(*(worker() for _ in range(args.workers)))
        finally:
            await browser.close()
            dataset.export()
