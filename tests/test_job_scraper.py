"""Tests for job_scraper service."""

from datetime import datetime, timezone

import pytest
import httpx
from unittest.mock import MagicMock, Mock, patch

from hr_breaker.services.job_scraper import (
    scrape_job_posting,
    CloudflareBlockedError,
    ScrapingError,
)
from hr_breaker.services.scrapers.base import BaseScraper
from hr_breaker.services.scrapers.httpx_scraper import HttpxScraper


class TestIsCloudflareBlocked:
    """Tests for Cloudflare detection."""

    def setup_method(self):
        self.scraper = HttpxScraper()

    def test_detects_cf_chl_opt(self):
        html = '<script>window._cf_chl_opt = {}</script>'
        assert self.scraper.is_cloudflare_blocked(html) is True

    def test_detects_challenge_platform(self):
        html = '<script src="/cdn-cgi/challenge-platform/something"></script>'
        assert self.scraper.is_cloudflare_blocked(html) is True

    def test_detects_just_a_moment(self):
        html = '<title>Just a moment...</title>'
        assert self.scraper.is_cloudflare_blocked(html) is True

    def test_detects_browser_verification(self):
        html = '<div id="cf-browser-verification">Verifying...</div>'
        assert self.scraper.is_cloudflare_blocked(html) is True

    def test_detects_checking_browser(self):
        html = '<p>Checking your browser before accessing</p>'
        assert self.scraper.is_cloudflare_blocked(html) is True

    def test_normal_page_not_blocked(self):
        html = '<html><body><h1>Job Posting</h1><p>Description here</p></body></html>'
        assert self.scraper.is_cloudflare_blocked(html) is False

    def test_empty_html_not_blocked(self):
        assert self.scraper.is_cloudflare_blocked('') is False


class TestHttpxScraper:
    """Tests for HttpxScraper."""

    def test_raises_cloudflare_error_on_challenge(self):
        cloudflare_html = '''
        <html>
        <head><title>Just a moment...</title></head>
        <body>
        <script>window._cf_chl_opt = {cvId: "3"}</script>
        </body>
        </html>
        '''
        mock_response = Mock()
        mock_response.text = cloudflare_html
        mock_response.status_code = 403

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            scraper = HttpxScraper(max_retries=1)
            with pytest.raises(CloudflareBlockedError) as exc_info:
                scraper.scrape('https://example.com/job')

            assert 'Cloudflare' in str(exc_info.value)

    def test_raises_scraping_error_on_non_cloudflare_403(self):
        html = '<html><body>Access Denied</body></html>'
        mock_response = Mock()
        mock_response.text = html
        mock_response.status_code = 403
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                'Forbidden', request=Mock(), response=mock_response
            )
        )

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            scraper = HttpxScraper(max_retries=1)
            with patch.object(scraper, '_backoff'):
                with pytest.raises(ScrapingError):
                    scraper.scrape('https://example.com/job')

    def test_extracts_job_content_from_article(self):
        html = '''
        <html>
        <body>
        <nav>Navigation</nav>
        <article>
            <h1>Software Engineer</h1>
            <p>We are looking for a talented engineer to join our team.</p>
            <p>Requirements: Python, JavaScript, SQL</p>
            <p>Benefits: Remote work, competitive salary</p>
        </article>
        <footer>Copyright 2025</footer>
        </body>
        </html>
        '''
        mock_response = Mock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            scraper = HttpxScraper()
            result = scraper.scrape('https://example.com/job')

            assert 'Software Engineer' in result
            assert 'Python' in result
            assert 'Navigation' not in result
            assert 'Copyright' not in result

    def test_extracts_job_content_from_job_div(self):
        html = '''
        <html>
        <body>
        <div class="job-description">
            <h2>Data Scientist</h2>
            <p>Join our data team!</p>
            <p>Skills: Machine Learning, Statistics, Python</p>
        </div>
        </body>
        </html>
        '''
        mock_response = Mock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            scraper = HttpxScraper()
            result = scraper.scrape('https://example.com/job')

            assert 'Data Scientist' in result
            assert 'Machine Learning' in result


class TestScrapeJobPosting:
    """Tests for scrape_job_posting function with fallback chain."""

    def test_raises_cloudflare_error_without_retry(self):
        """CloudflareBlockedError should not trigger retries within httpx scraper."""
        cloudflare_html = '<script>window._cf_chl_opt = {}</script>'
        mock_response = Mock()
        mock_response.text = cloudflare_html
        mock_response.status_code = 403

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get = mock_get

            # Disable fallbacks to test httpx behavior alone
            with pytest.raises(ScrapingError):
                scrape_job_posting(
                    'https://example.com/job',
                    max_retries=3,
                    use_wayback=False,
                    use_playwright=False,
                )

            # Should only try once - no retries for Cloudflare
            assert call_count == 1

    def test_retries_on_non_cloudflare_403(self):
        """Non-Cloudflare 403 should trigger retries."""
        html = '<html>Forbidden</html>'
        mock_response = Mock()
        mock_response.text = html
        mock_response.status_code = 403
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                'Forbidden', request=Mock(), response=mock_response
            )
        )

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get = mock_get

            with patch('hr_breaker.services.scrapers.httpx_scraper.time.sleep'):
                with pytest.raises(ScrapingError):
                    scrape_job_posting(
                        'https://example.com/job',
                        max_retries=3,
                        use_wayback=False,
                        use_playwright=False,
                    )

            # Should try 3 times
            assert call_count == 3

    def test_returns_content_on_success(self):
        html = '''
        <html><body>
        <article>
            <h1>Great Job</h1>
            <p>This is a great opportunity with lots of details.</p>
            <p>More content here to make it long enough.</p>
        </article>
        </body></html>
        '''
        mock_response = Mock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as mock_client:
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            result = scrape_job_posting('https://example.com/job')

            assert 'Great Job' in result

    def test_skips_wayback_on_cloudflare(self):
        """Wayback should be skipped when httpx fails with Cloudflare (optimization)."""
        cloudflare_html = '<script>window._cf_chl_opt = {}</script>'

        httpx_response = Mock()
        httpx_response.text = cloudflare_html
        httpx_response.status_code = 403

        httpx_call_count = 0
        wayback_call_count = 0

        def mock_get(url, **kwargs):
            nonlocal httpx_call_count, wayback_call_count
            # Count calls based on URL pattern
            if 'web.archive.org' in url:
                wayback_call_count += 1
                return Mock(json=lambda: [], text='', raise_for_status=Mock())
            else:
                httpx_call_count += 1
                return httpx_response

        # Single patch - both modules share the same httpx
        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as client_mock:
            client_mock.return_value.__enter__.return_value.get = mock_get

            with pytest.raises(ScrapingError):
                scrape_job_posting(
                    'https://example.com/job',
                    use_wayback=True,
                    use_playwright=False,
                )

            # httpx should be called once (cloudflare detected)
            assert httpx_call_count == 1
            # Wayback should NOT be called for Cloudflare-blocked sites
            assert wayback_call_count == 0

    def test_fallback_to_wayback_on_non_cloudflare_error(self):
        """Should try Wayback when httpx fails with non-Cloudflare error."""
        error_html = '<html>Server Error</html>'
        wayback_html = '''
        <html><body>
        <article>
            <h1>Archived Job</h1>
            <p>This is job content from the archive with enough text.</p>
        </article>
        </body></html>
        '''

        httpx_response = Mock()
        httpx_response.text = error_html
        httpx_response.status_code = 500
        httpx_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                'Server Error', request=Mock(), response=httpx_response
            )
        )

        wayback_response = Mock()
        wayback_response.text = wayback_html
        wayback_response.status_code = 200
        wayback_response.raise_for_status = Mock()

        # CDX API response
        cdx_response = Mock()
        cdx_response.status_code = 200
        cdx_response.json = Mock(return_value=[
            ["urlkey", "timestamp", "original", "mimetype", "statuscode", "digest", "length"],
            ["com,example)/job", datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"), "https://example.com/job", "text/html", "200", "abc", "1000"]
        ])
        cdx_response.raise_for_status = Mock()

        def mock_wayback_get(url, **kwargs):
            if 'web.archive.org/cdx' in url:
                return cdx_response
            if 'web.archive.org/web' in url:
                return wayback_response
            return httpx_response

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as httpx_mock:
            httpx_mock.return_value.__enter__.return_value.get.return_value = httpx_response

            with patch('hr_breaker.services.scrapers.wayback_scraper.httpx.Client') as wayback_mock:
                wayback_mock.return_value.__enter__.return_value.get = mock_wayback_get

                result = scrape_job_posting(
                    'https://example.com/job',
                    use_wayback=True,
                    use_playwright=False,
                )

                assert 'Archived Job' in result

    def test_error_includes_all_methods_tried(self):
        """Error message should list all fallback methods attempted."""
        # Use non-cloudflare error so wayback is attempted
        error_html = '<html>Server Error</html>'

        # Create simple response objects that behave predictably
        class MockResponse:
            def __init__(self, text, status_code):
                self.text = text
                self.status_code = status_code

            def json(self):
                return []

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        'Error', request=Mock(), response=self
                    )

        httpx_response = MockResponse(error_html, 500)
        cdx_response = MockResponse("", 200)  # Empty CDX response returns []
        cdx_response.json = lambda: []  # Override json method

        def mock_get(url, **kwargs):
            if 'web.archive.org/cdx' in url:
                return cdx_response
            return httpx_response

        with patch('hr_breaker.services.scrapers.httpx_scraper.httpx.Client') as httpx_mock:
            httpx_mock.return_value.__enter__.return_value.get = mock_get

            with patch('hr_breaker.services.scrapers.wayback_scraper.httpx.Client') as wayback_mock:
                wayback_mock.return_value.__enter__.return_value.get = mock_get

                with pytest.raises(ScrapingError) as exc_info:
                    scrape_job_posting(
                        'https://example.com/job',
                        use_wayback=True,
                        use_playwright=False,
                    )

                error_msg = str(exc_info.value)
                assert 'httpx' in error_msg
                assert 'wayback' in error_msg
