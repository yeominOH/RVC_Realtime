#include "AraRvcDocumentController.h"
#include "AraRvcPlaybackRenderer.h"

juce::ARAPlaybackRenderer* AraRvcDocumentController::doCreatePlaybackRenderer()
{
    return new AraRvcPlaybackRenderer(getDocumentController());
}

bool AraRvcDocumentController::doRestoreObjectsFromStream(juce::ARAInputStream& input,
                                                          const juce::ARARestoreObjectsFilter* filter)
{
    juce::ignoreUnused(input, filter);
    return true;
}

bool AraRvcDocumentController::doStoreObjectsToStream(juce::ARAOutputStream& output,
                                                      const juce::ARAStoreObjectsFilter* filter)
{
    juce::ignoreUnused(output, filter);
    return true;
}

const ARA::ARAFactory* JUCE_CALLTYPE createARAFactory()
{
    return juce::ARADocumentControllerSpecialisation::createARAFactory<AraRvcDocumentController>();
}
