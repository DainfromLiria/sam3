# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import inspect
import types
import unittest
from functools import wraps
from unittest.mock import MagicMock, patch

import torch
from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

# The multiplex module checks GPU hardware at import. Test its real API on CPU.
with patch("torch.cuda.get_device_properties", return_value=types.SimpleNamespace(major=7)):
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity


class _StreamingModel:
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        del start_frame_idx, max_frame_num_to_track, reverse
        for frame_idx in range(2):
            inference_state["cached_frame_outputs"][frame_idx] = {
                7: torch.ones((1, 4, 4))
            }
            yield frame_idx, {"frame": frame_idx}


class _SessionModel:
    def __init__(self, init_state_method):
        # Keep the real model signature without allocating its GPU weights.
        @wraps(init_state_method)
        def init_state(model, *args, **kwargs):
            bound = inspect.signature(init_state_method).bind(model, *args, **kwargs)
            model.received = bound.arguments
            return {"num_frames": 1}

        self.init_state = types.MethodType(init_state, self)


class TestVideoPredictorSession(unittest.TestCase):
    def _predictor(self, init_state_method):
        predictor = Sam3BasePredictor()
        predictor.model = _SessionModel(init_state_method)
        predictor.async_loading_frames = True
        predictor.video_loader_type = "cv2"
        return predictor

    def test_sam31_session_accepts_supported_options(self):
        predictor = self._predictor(Sam3MultiplexTrackingWithInteractivity.init_state)
        response = predictor.handle_request({
            "type": "start_session", "resource_path": "/frames",
            "offload_video_to_cpu": True,
        })
        received = predictor.model.received
        self.assertEqual("/frames", received["resource_path"])
        self.assertTrue(received["offload_video_to_cpu"])
        self.assertTrue(received["async_loading_frames"])
        self.assertNotIn("offload_state_to_cpu", received)
        self.assertNotIn("video_loader_type", received)
        self.assertIn(response["session_id"], predictor._all_inference_states)

    def test_sam3_session_preserves_state_offload(self):
        predictor = self._predictor(Sam3VideoInferenceWithInstanceInteractivity.init_state)
        predictor.handle_request({
            "type": "start_session", "resource_path": "/frames",
            "offload_state_to_cpu": True,
        })
        self.assertTrue(predictor.model.received["offload_state_to_cpu"])
        self.assertEqual("cv2", predictor.model.received["video_loader_type"])

    def test_sam31_rejects_requested_state_offload(self):
        predictor = self._predictor(Sam3MultiplexTrackingWithInteractivity.init_state)
        with self.assertRaisesRegex(ValueError, "offload_state_to_cpu"):
            predictor.handle_request({
                "type": "start_session", "resource_path": "/frames",
                "offload_state_to_cpu": True,
            })
        self.assertEqual({}, predictor._all_inference_states)


class TestVideoPredictorStreaming(unittest.TestCase):
    def _predictor(self):
        predictor = Sam3BasePredictor()
        predictor.model = _StreamingModel()
        state = {"cached_frame_outputs": {}}
        predictor._all_inference_states["test"] = {
            "state": state,
            "last_use_time": 0,
        }
        return predictor, state

    def test_streaming_eviction_releases_each_yielded_frame(self) -> None:
        predictor, state = self._predictor()
        responses = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": "test",
                "propagation_direction": "forward",
                "evict_cached_frame_outputs": True,
            }
        )

        self.assertEqual(0, next(responses)["frame_index"])
        self.assertNotIn(0, state["cached_frame_outputs"])
        self.assertEqual(1, next(responses)["frame_index"])
        self.assertNotIn(1, state["cached_frame_outputs"])

    def test_streaming_cache_is_retained_by_default(self) -> None:
        predictor, state = self._predictor()
        list(
            predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": "test",
                    "propagation_direction": "forward",
                }
            )
        )

        self.assertEqual([0, 1], sorted(state["cached_frame_outputs"]))


class TestVideoPredictorShutdown(unittest.TestCase):
    def test_exits_tracker_autocast_context(self) -> None:
        predictor = Sam3VideoPredictor.__new__(Sam3VideoPredictor)
        context = torch.amp.autocast("cpu", dtype=torch.bfloat16)
        context.__enter__()
        tracker = types.SimpleNamespace(bf16_context=context)
        predictor.model = types.SimpleNamespace(tracker=tracker)
        predictor._all_inference_states = {}
        self.assertTrue(torch.is_autocast_enabled("cpu"))

        predictor.shutdown()
        predictor.shutdown()

        self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertIsNone(tracker.bf16_context)


class TestRemoveObjectRequest(unittest.TestCase):
    def test_forwards_is_user_action(self) -> None:
        predictor = Sam3VideoPredictor.__new__(Sam3VideoPredictor)
        predictor.remove_object = MagicMock()
        request = {"type": "remove_object", "session_id": "s", "obj_id": 3}

        predictor.handle_request({**request, "is_user_action": False})
        predictor.handle_request(request)

        self.assertEqual(
            [
                c.kwargs["is_user_action"]
                for c in predictor.remove_object.call_args_list
            ],
            [False, True],
        )


if __name__ == "__main__":
    unittest.main()
