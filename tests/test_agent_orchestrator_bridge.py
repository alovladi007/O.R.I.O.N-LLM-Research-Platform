"""
Integration Tests for Agent-Orchestrator Bridge
=================================================

Tests the integration between the Agent API and Orchestrator/Campaign system:
- Campaign creation via agent API
- Campaign advancement via agent API
- Design iteration execution
- Result tracking
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models.user import User
from src.api.models.campaign import DesignCampaign, DesignIteration, CampaignStatus
from src.api.models.structure import Structure, StructureSource
from src.api.models.material import Material
from src.api.schemas.orchestrator import (
    AgentCreateCampaignRequest,
    AgentAdvanceCampaignRequest,
)


class TestAgentOrchestratorBridge:
    """Test agent-orchestrator bridge integration"""

    @pytest.mark.asyncio
    async def test_create_campaign_via_agent_api(self, db_session: AsyncSession, test_user: User):
        """Test creating a design campaign via agent API"""
        from src.api.routers.agent import create_design_campaign

        # Create request
        request = AgentCreateCampaignRequest(
            name="Test Campaign",
            goal="Find 2eV bandgap materials",
            target_properties={
                "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0}
            },
            constraints={"elements": ["Mo", "S", "Se"]},
            max_iterations=5
        )

        # Create campaign
        response = await create_design_campaign(
            request=request,
            db=db_session,
            current_user=test_user
        )

        # Verify response
        assert response.success is True
        assert response.command_type == "create_design_campaign"
        assert "campaign_id" in response.result

        # Verify campaign in DB
        campaign_id = uuid.UUID(response.result["campaign_id"])
        campaign = await db_session.get(DesignCampaign, campaign_id)

        assert campaign is not None
        assert campaign.name == "Test Campaign"
        assert campaign.status == CampaignStatus.PENDING
        assert campaign.max_iterations == 5
        assert campaign.current_iteration == 0
        assert campaign.config["target_properties"]["bandgap"]["value"] == 2.0

    @pytest.mark.asyncio
    async def test_advance_campaign_activates_pending_campaign(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test that advancing a PENDING campaign activates it"""
        # Create a PENDING campaign
        campaign = DesignCampaign(
            owner_id=test_user.id,
            name="Test Campaign",
            description="Test",
            status=CampaignStatus.PENDING,
            max_iterations=3,
            current_iteration=0,
            config={
                "target_properties": {"bandgap": {"value": 2.0}},
                "candidates_per_iteration": 2
            }
        )
        db_session.add(campaign)
        await db_session.commit()
        await db_session.refresh(campaign)

        # Mock the DesignLoopService to avoid actual structure generation
        with patch('src.api.routers.agent.DesignLoopService') as mock_service:
            # Create mock iteration
            mock_iteration = Mock()
            mock_iteration.iteration_index = 0
            mock_iteration.best_score_this_iter = 0.85
            mock_iteration.num_candidates_created = 2
            mock_iteration.metrics = {"mean_score": 0.80}

            mock_service.run_iteration = AsyncMock(return_value=mock_iteration)

            from src.api.routers.agent import advance_campaign

            # Create advancement request
            request = AgentAdvanceCampaignRequest(
                campaign_id=campaign.id,
                num_iterations=1
            )

            # Advance campaign
            response = await advance_campaign(
                request=request,
                db=db_session,
                current_user=test_user
            )

        # Verify response
        assert response.success is True
        assert response.result["iterations_completed"] == 1

        # Verify campaign status changed to RUNNING
        await db_session.refresh(campaign)
        assert campaign.status == CampaignStatus.RUNNING
        assert campaign.started_at is not None

    @pytest.mark.asyncio
    async def test_advance_campaign_runs_multiple_iterations(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test advancing campaign with multiple iterations"""
        # Create a RUNNING campaign
        campaign = DesignCampaign(
            owner_id=test_user.id,
            name="Multi-Iteration Campaign",
            status=CampaignStatus.RUNNING,
            started_at=datetime.utcnow(),
            max_iterations=10,
            current_iteration=0,
            config={
                "target_properties": {"bandgap": {"value": 2.0}},
                "candidates_per_iteration": 2
            }
        )
        db_session.add(campaign)
        await db_session.commit()
        await db_session.refresh(campaign)

        # Mock the DesignLoopService
        iteration_count = 0

        async def mock_run_iteration(db, campaign_id):
            nonlocal iteration_count
            mock_iter = Mock()
            mock_iter.iteration_index = iteration_count
            mock_iter.best_score_this_iter = 0.8 + (iteration_count * 0.02)
            mock_iter.num_candidates_created = 2
            mock_iter.metrics = {"mean_score": 0.75 + (iteration_count * 0.02)}
            iteration_count += 1

            # Update campaign iteration in mock
            campaign.current_iteration = iteration_count
            return mock_iter

        with patch('src.api.routers.agent.DesignLoopService') as mock_service:
            mock_service.run_iteration = mock_run_iteration

            from src.api.routers.agent import advance_campaign

            # Request 3 iterations
            request = AgentAdvanceCampaignRequest(
                campaign_id=campaign.id,
                num_iterations=3
            )

            response = await advance_campaign(
                request=request,
                db=db_session,
                current_user=test_user
            )

        # Verify all iterations completed
        assert response.success is True
        assert response.result["iterations_completed"] == 3
        assert response.result["iterations_requested"] == 3
        assert len(response.result["iterations"]) == 3

        # Verify scores increased
        scores = [iter["best_score"] for iter in response.result["iterations"]]
        assert scores[0] < scores[1] < scores[2]

    @pytest.mark.asyncio
    async def test_advance_campaign_handles_completion(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test that campaign completes when reaching max iterations"""
        # Create campaign near completion
        campaign = DesignCampaign(
            owner_id=test_user.id,
            name="Near Complete Campaign",
            status=CampaignStatus.RUNNING,
            started_at=datetime.utcnow(),
            max_iterations=3,
            current_iteration=2,  # One iteration left
            config={
                "target_properties": {"bandgap": {"value": 2.0}},
                "candidates_per_iteration": 2
            }
        )
        db_session.add(campaign)
        await db_session.commit()
        await db_session.refresh(campaign)

        async def mock_run_iteration(db, campaign_id):
            mock_iter = Mock()
            mock_iter.iteration_index = 2
            mock_iter.best_score_this_iter = 0.95
            mock_iter.num_candidates_created = 2
            mock_iter.metrics = {"mean_score": 0.92}

            # Mark campaign as completed
            campaign.current_iteration = 3
            campaign.status = CampaignStatus.COMPLETED
            campaign.completed_at = datetime.utcnow()
            return mock_iter

        with patch('src.api.routers.agent.DesignLoopService') as mock_service:
            mock_service.run_iteration = mock_run_iteration

            from src.api.routers.agent import advance_campaign

            request = AgentAdvanceCampaignRequest(
                campaign_id=campaign.id,
                num_iterations=5  # Request more than available
            )

            response = await advance_campaign(
                request=request,
                db=db_session,
                current_user=test_user
            )

        # Verify only 1 iteration ran (reached max)
        assert response.success is True
        assert response.result["iterations_completed"] == 1
        assert response.result["campaign_status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_advance_campaign_validation(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test validation for campaign advancement"""
        from src.api.routers.agent import advance_campaign
        from fastapi import HTTPException

        # Test: Campaign not found
        request = AgentAdvanceCampaignRequest(
            campaign_id=uuid.uuid4(),  # Non-existent
            num_iterations=1
        )

        with pytest.raises(HTTPException) as exc_info:
            await advance_campaign(
                request=request,
                db=db_session,
                current_user=test_user
            )

        assert exc_info.value.status_code == 404

        # Test: Campaign already completed
        completed_campaign = DesignCampaign(
            owner_id=test_user.id,
            name="Completed Campaign",
            status=CampaignStatus.COMPLETED,
            max_iterations=5,
            current_iteration=5,
            config={}
        )
        db_session.add(completed_campaign)
        await db_session.commit()
        await db_session.refresh(completed_campaign)

        request = AgentAdvanceCampaignRequest(
            campaign_id=completed_campaign.id,
            num_iterations=1
        )

        with pytest.raises(HTTPException) as exc_info:
            await advance_campaign(
                request=request,
                db=db_session,
                current_user=test_user
            )

        assert exc_info.value.status_code == 400
        assert "already completed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_advance_campaign_handles_iteration_failures(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test that advancement handles iteration failures gracefully"""
        campaign = DesignCampaign(
            owner_id=test_user.id,
            name="Failure Test Campaign",
            status=CampaignStatus.RUNNING,
            started_at=datetime.utcnow(),
            max_iterations=5,
            current_iteration=0,
            config={"candidates_per_iteration": 2}
        )
        db_session.add(campaign)
        await db_session.commit()
        await db_session.refresh(campaign)

        iteration_count = 0

        async def mock_run_iteration_with_failure(db, campaign_id):
            nonlocal iteration_count
            iteration_count += 1

            if iteration_count == 2:
                # Second iteration fails
                raise ValueError("Simulated iteration failure")

            # Other iterations succeed
            mock_iter = Mock()
            mock_iter.iteration_index = iteration_count - 1
            mock_iter.best_score_this_iter = 0.80
            mock_iter.num_candidates_created = 2
            mock_iter.metrics = {"mean_score": 0.75}

            campaign.current_iteration = iteration_count
            return mock_iter

        with patch('src.api.routers.agent.DesignLoopService') as mock_service:
            mock_service.run_iteration = mock_run_iteration_with_failure

            from src.api.routers.agent import advance_campaign

            request = AgentAdvanceCampaignRequest(
                campaign_id=campaign.id,
                num_iterations=3
            )

            response = await advance_campaign(
                request=request,
                db=db_session,
                current_user=test_user
            )

        # Verify: 2 succeeded, 1 failed
        assert response.success is True
        assert response.result["iterations_completed"] == 2
        assert response.result["iterations_failed"] == 1
        assert "failed_iterations" in response.result
        assert len(response.result["failed_iterations"]) == 1


class TestDesignLoopServiceStub:
    """Test that DesignLoopService can be imported and has correct interface"""

    def test_design_loop_service_exists(self):
        """Test that DesignLoopService exists and has required methods"""
        from backend.common.campaigns.loop import DesignLoopService

        # Verify class exists
        assert DesignLoopService is not None

        # Verify required methods exist
        assert hasattr(DesignLoopService, 'run_iteration')
        assert hasattr(DesignLoopService, 'generate_candidates')
        assert hasattr(DesignLoopService, 'evaluate_candidates')
        assert hasattr(DesignLoopService, 'calculate_score')


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
