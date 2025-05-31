from abc import ABC, abstractmethod
from imagai.models import ImageGenerationRequest, ImageGenerationResponse
from typing import List


class BaseImageProvider(ABC):
    @abstractmethod
    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> List[ImageGenerationResponse]:
        """
        Generates an image based on the provided request.
        Returns a list of responses, one for each image generated (n).
        """
        pass
