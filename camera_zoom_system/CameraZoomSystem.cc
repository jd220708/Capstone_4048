#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <gz/common/Console.hh>
#include <gz/common/Event.hh>

#include <gz/math/Angle.hh>

#include <gz/msgs/double.pb.h>

#include <gz/plugin/Register.hh>

#include <gz/rendering/Camera.hh>
#include <gz/rendering/RenderEngine.hh>
#include <gz/rendering/RenderingIface.hh>
#include <gz/rendering/Scene.hh>
#include <gz/rendering/Sensor.hh>

#include <gz/sim/EventManager.hh>
#include <gz/sim/System.hh>
#include <gz/sim/rendering/Events.hh>

#include <gz/transport/Node.hh>


namespace camera_zoom
{

class CameraZoomSystem:
  public gz::sim::System,
  public gz::sim::ISystemConfigure
{
  public:

    void Configure(
      const gz::sim::Entity &,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &,
      gz::sim::EventManager &_eventMgr
    ) override
    {
      // ---------------------------------------------
      // Configuration
      // ---------------------------------------------

      if (_sdf->HasElement("command_topic"))
      {
        this->commandTopic =
          _sdf->Get<std::string>("command_topic");
      }

      if (_sdf->HasElement("camera_name_contains"))
      {
        this->cameraNameContains =
          _sdf->Get<std::string>("camera_name_contains");
      }

      if (_sdf->HasElement("min_zoom"))
      {
        this->minZoom =
          _sdf->Get<double>("min_zoom");
      }

      if (_sdf->HasElement("max_zoom"))
      {
        this->maxZoom =
          _sdf->Get<double>("max_zoom");
      }


      // ---------------------------------------------
      // Gazebo zoom command subscription
      // ---------------------------------------------

      const bool subscribed =
        this->node.Subscribe(
          this->commandTopic,
          &CameraZoomSystem::OnZoomCommand,
          this
        );

      if (!subscribed)
      {
        gzerr
          << "[CameraZoom] Could not subscribe to ["
          << this->commandTopic
          << "]\n";
      }
      else
      {
        gzmsg
          << "[CameraZoom] Listening on ["
          << this->commandTopic
          << "]\n";
      }


      // ---------------------------------------------
      // Rendering-thread callback
      // ---------------------------------------------

      this->renderConnection =
        _eventMgr.Connect<gz::sim::events::PreRender>(
          std::bind(
            &CameraZoomSystem::OnPreRender,
            this
          )
        );
    }


  private:

    void OnZoomCommand(
      const gz::msgs::Double &_msg
    )
    {
      const double requested =
        std::clamp(
          _msg.data(),
          this->minZoom,
          this->maxZoom
        );

      {
        std::lock_guard<std::mutex> lock(
          this->zoomMutex
        );

        this->requestedZoom = requested;
        this->zoomDirty = true;
      }

      gzmsg
        << "[CameraZoom] Requested zoom: "
        << requested
        << "x\n";
    }


  private:

    void FindScene()
    {
      auto engines =
        gz::rendering::loadedEngines();

      if (engines.empty())
      {
        return;
      }

      auto engine =
        gz::rendering::engine(
          engines.front()
        );

      if (!engine)
      {
        return;
      }

      if (engine->SceneCount() == 0)
      {
        return;
      }

      auto scene =
        engine->SceneByIndex(0);

      if (!scene)
      {
        return;
      }

      if (
        !scene->IsInitialized() ||
        !scene->RootVisual()
      )
      {
        return;
      }

      this->scene = scene;

      gzmsg
        << "[CameraZoom] Rendering scene found: ["
        << this->scene->Name()
        << "]\n";
    }


  private:

    void FindCamera()
    {
      if (!this->scene)
      {
        return;
      }

      for (
        unsigned int i = 0;
        i < this->scene->SensorCount();
        ++i
      )
      {
        auto sensor =
          this->scene->SensorByIndex(i);

        if (!sensor)
        {
          continue;
        }

        auto camera =
          std::dynamic_pointer_cast<
            gz::rendering::Camera
          >(sensor);

        if (!camera)
        {
          continue;
        }

        gzmsg
          << "[CameraZoom] Camera candidate: ["
          << camera->Name()
          << "]\n";


        // Pick camera whose rendering name contains our
        // configured string.
        if (
          camera->Name().find(
            this->cameraNameContains
          ) != std::string::npos
        )
        {
          this->camera = camera;

          this->referenceHFOV =
            this->camera->HFOV().Radian();

          gzmsg
            << "[CameraZoom] Selected camera: ["
            << this->camera->Name()
            << "]\n";

          gzmsg
            << "[CameraZoom] Base HFOV: "
            << this->referenceHFOV
            << " rad\n";

          return;
        }
      }
    }


  private:

    void OnPreRender()
    {
      // ---------------------------------------------
      // Find rendering scene
      // ---------------------------------------------

      if (!this->scene)
      {
        this->FindScene();
      }

      if (!this->scene)
      {
        return;
      }


      // ---------------------------------------------
      // Find camera
      // ---------------------------------------------

      if (!this->camera)
      {
        this->FindCamera();
      }

      if (!this->camera)
      {
        return;
      }


      // ---------------------------------------------
      // Get requested zoom
      // ---------------------------------------------

      double zoom = 1.0;
      bool dirty = false;

      {
        std::lock_guard<std::mutex> lock(
          this->zoomMutex
        );

        zoom = this->requestedZoom;
        dirty = this->zoomDirty;

        this->zoomDirty = false;
      }

      if (!dirty)
      {
        return;
      }


      // ---------------------------------------------
      // Convert optical zoom factor to HFOV
      //
      // zoom = focal length multiplier
      //
      // HFOV_new =
      // 2 atan(tan(HFOV_original/2) / zoom)
      // ---------------------------------------------

      const double newHFOV =
        2.0 *
        std::atan(
          std::tan(
            this->referenceHFOV / 2.0
          ) /
          zoom
        );


      gz::math::Angle angle;

      angle.SetRadian(
        newHFOV
      );


      // ---------------------------------------------
      // Apply actual rendering-camera zoom
      // ---------------------------------------------

      this->camera->SetHFOV(
        angle
      );


      gzmsg
        << "[CameraZoom] Zoom = "
        << zoom
        << "x | HFOV = "
        << newHFOV
        << " rad\n";
    }


  // =========================================================
  // SETTINGS
  // =========================================================

  private:

    std::string commandTopic{
      "/model/x500_gimbal_0/camera/zoom/cmd_zoom"
    };

    std::string cameraNameContains{
      "camera"
    };

    double minZoom{1.0};

    double maxZoom{8.0};


  // =========================================================
  // RENDERING
  // =========================================================

  private:

    gz::rendering::ScenePtr scene;

    gz::rendering::CameraPtr camera;

    double referenceHFOV{-1.0};

    gz::common::ConnectionPtr renderConnection;


  // =========================================================
  // COMMAND STATE
  // =========================================================

  private:

    gz::transport::Node node;

    std::mutex zoomMutex;

    double requestedZoom{1.0};

    bool zoomDirty{true};
};

}


// ============================================================
// REGISTER PLUGIN
// ============================================================

GZ_ADD_PLUGIN(
  camera_zoom::CameraZoomSystem,
  gz::sim::System,
  camera_zoom::CameraZoomSystem::ISystemConfigure
)

GZ_ADD_PLUGIN_ALIAS(
  camera_zoom::CameraZoomSystem,
  "camera_zoom::CameraZoomSystem"
)
