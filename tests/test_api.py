"""
API Tests
=========

Comprehensive test suite for the ORION API.
"""

import pytest
import httpx
from datetime import datetime, timedelta
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.config import settings
from src.api.auth.security import SecurityService
from src.api.models import User, Material, Simulation
from src.api.schemas.auth import UserCreate


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    @pytest.mark.asyncio
    async def test_register_user(self, async_client: httpx.AsyncClient):
        """Test user registration"""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "SecurePassword123!",
            "full_name": "Test User"
        }
        
        response = await async_client.post(
            "/api/v1/auth/register",
            json=user_data
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == user_data["email"]
        assert data["username"] == user_data["username"]
        assert "id" in data
        assert "password" not in data
    
    @pytest.mark.asyncio
    async def test_login_success(self, async_client: httpx.AsyncClient, test_user: User):
        """Test successful login"""
        response = await async_client.post(
            "/api/v1/auth/token",
            data={
                "username": test_user.email,
                "password": "testpassword123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, async_client: httpx.AsyncClient):
        """Test login with invalid credentials"""
        response = await async_client.post(
            "/api/v1/auth/token",
            data={
                "username": "invalid@example.com",
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect username or password"
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client: httpx.AsyncClient, test_user: User):
        """Test token refresh"""
        # First, login to get tokens
        login_response = await async_client.post(
            "/api/v1/auth/token",
            data={
                "username": test_user.email,
                "password": "testpassword123"
            }
        )
        
        refresh_token = login_response.json()["refresh_token"]
        
        # Use refresh token
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    @pytest.mark.asyncio
    async def test_get_current_user(
        self,
        async_client: httpx.AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test getting current user info"""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_user.id)
        assert data["email"] == test_user.email
        assert data["username"] == test_user.username


class TestUserEndpoints:
    """Test user management endpoints"""
    
    @pytest.mark.asyncio
    async def test_list_users_as_admin(
        self,
        async_client: httpx.AsyncClient,
        admin_headers: dict
    ):
        """Test listing users as admin"""
        response = await async_client.get(
            "/api/v1/users",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
    
    @pytest.mark.asyncio
    async def test_list_users_as_regular_user(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict
    ):
        """Test listing users as regular user (should fail)"""
        response = await async_client.get(
            "/api/v1/users",
            headers=auth_headers
        )
        
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_update_user_profile(
        self,
        async_client: httpx.AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test updating user profile"""
        update_data = {
            "full_name": "Updated Name",
            "bio": "New bio",
            "organization": "Test Org"
        }
        
        response = await async_client.patch(
            f"/api/v1/users/{test_user.id}",
            json=update_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == update_data["full_name"]
        assert data["bio"] == update_data["bio"]
        assert data["organization"] == update_data["organization"]
    
    @pytest.mark.asyncio
    async def test_change_password(
        self,
        async_client: httpx.AsyncClient,
        test_user: User,
        auth_headers: dict
    ):
        """Test changing password"""
        response = await async_client.post(
            "/api/v1/users/change-password",
            json={
                "current_password": "testpassword123",
                "new_password": "NewSecurePassword123!"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert response.json()["message"] == "Password changed successfully"


class TestMaterialsEndpoints:
    """Test materials endpoints"""
    
    @pytest.mark.asyncio
    async def test_create_material(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict
    ):
        """Test creating a new material"""
        material_data = {
            "formula": "TiO2",
            "name": "Titanium Dioxide",
            "description": "Rutile phase titanium dioxide",
            "properties": {
                "bandgap": 3.0,
                "density": 4.23,
                "crystal_system": "tetragonal"
            }
        }
        
        response = await async_client.post(
            "/api/v1/materials",
            json=material_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["formula"] == material_data["formula"]
        assert data["name"] == material_data["name"]
        assert "id" in data
        assert "created_at" in data
    
    @pytest.mark.asyncio
    async def test_search_materials(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict,
        test_materials: list[Material]
    ):
        """Test searching materials"""
        response = await async_client.get(
            "/api/v1/materials/search",
            params={
                "query": "oxide",
                "limit": 10
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert len(data["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_material_by_id(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict,
        test_material: Material
    ):
        """Test getting material by ID"""
        response = await async_client.get(
            f"/api/v1/materials/{test_material.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_material.id)
        assert data["formula"] == test_material.formula
    
    @pytest.mark.asyncio
    async def test_generate_material_candidates(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict
    ):
        """Test AI-powered material generation"""
        response = await async_client.post(
            "/api/v1/materials/generate",
            json={
                "description": "High temperature superconductor with Tc > 100K",
                "constraints": {
                    "elements": ["Cu", "O", "Y", "Ba"],
                    "max_elements": 4
                },
                "num_candidates": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "candidates" in data
        assert len(data["candidates"]) <= 5
        assert all("formula" in c for c in data["candidates"])
        assert all("score" in c for c in data["candidates"])


class TestSimulationEndpoints:
    """Test simulation endpoints"""
    
    @pytest.mark.asyncio
    async def test_submit_simulation(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict,
        test_material: Material
    ):
        """Test submitting a simulation job"""
        simulation_data = {
            "material_id": str(test_material.id),
            "simulation_type": "dft",
            "parameters": {
                "functional": "PBE",
                "cutoff_energy": 520,
                "k_points": [4, 4, 4]
            }
        }
        
        response = await async_client.post(
            "/api/v1/simulations",
            json=simulation_data,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["simulation_type"] == "dft"
    
    @pytest.mark.asyncio
    async def test_get_simulation_status(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict,
        test_simulation: Simulation
    ):
        """Test getting simulation status"""
        response = await async_client.get(
            f"/api/v1/simulations/{test_simulation.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_simulation.id)
        assert "status" in data
        assert "progress" in data
    
    @pytest.mark.asyncio
    async def test_list_user_simulations(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict
    ):
        """Test listing user's simulations"""
        response = await async_client.get(
            "/api/v1/simulations/my-simulations",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data


class TestWebSocketEndpoints:
    """Test WebSocket connections"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(
        self,
        websocket_client,
        auth_token: str
    ):
        """Test establishing WebSocket connection"""
        async with websocket_client.websocket_connect(
            f"/ws/notifications?token={auth_token}"
        ) as websocket:
            # Send a test message
            await websocket.send_json({
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            })
            
            # Receive response
            data = await websocket.receive_json()
            assert data["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_simulation_updates_via_websocket(
        self,
        websocket_client,
        auth_token: str,
        test_simulation: Simulation
    ):
        """Test receiving simulation updates via WebSocket"""
        async with websocket_client.websocket_connect(
            f"/ws/simulations/{test_simulation.id}?token={auth_token}"
        ) as websocket:
            # Should receive initial status
            data = await websocket.receive_json()
            assert data["type"] == "simulation_status"
            assert data["simulation_id"] == str(test_simulation.id)


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict
    ):
        """Test that rate limiting is enforced"""
        # Make requests up to the limit
        for _ in range(settings.rate_limit_requests):
            response = await async_client.get(
                "/api/v1/materials",
                headers=auth_headers
            )
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = await async_client.get(
            "/api/v1/materials",
            headers=auth_headers
        )
        
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
        assert "Retry-After" in response.headers


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_404_error(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict
    ):
        """Test 404 error handling"""
        response = await async_client.get(
            "/api/v1/materials/nonexistent-id",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "detail" in response.json()
    
    @pytest.mark.asyncio
    async def test_validation_error(
        self,
        async_client: httpx.AsyncClient,
        auth_headers: dict
    ):
        """Test validation error handling"""
        response = await async_client.post(
            "/api/v1/materials",
            json={
                "formula": "",  # Invalid: empty formula
                "name": "Test Material"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert any(e["loc"] == ["body", "formula"] for e in data["detail"])
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, async_client: httpx.AsyncClient):
        """Test unauthorized access"""
        response = await async_client.get("/api/v1/users")
        
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"