// ============================================================
// SPHERE NETWORK SYSTEM
// ============================================================
//
// A Gazebo System plugin for creating visual-only moving spheres.
//
// IMPORTANT TERMINOLOGY:
//
//   GROUP = one linear path / stream
//
// Example:
//
//   group_1:
//
//   START -------------------------------------> END
//
//        sphere
//           |
//           | random 1-4 sec
//           v
//        sphere
//           |
//           | random 1-4 sec
//           v
//        sphere
//
// Each sphere:
//
//   - spawns individually
//   - receives a random colour
//   - receives a slightly random speed
//   - moves directly along its group path
//   - has no collision
//   - has no physics behaviour
//   - is deleted when it reaches the path end
//
// All spawning, movement and deletion happens INSIDE Gazebo.
//
// No ROS service calls.
// No Gazebo Transport service calls.
// No VelocityControl plugin per sphere.
//
// ============================================================


#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>


#include <gz/common/Console.hh>

#include <gz/math/Color.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>

#include <gz/plugin/Register.hh>

#include <gz/sim/Entity.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/SdfEntityCreator.hh>
#include <gz/sim/System.hh>


#include <sdf/Element.hh>
#include <sdf/Geometry.hh>
#include <sdf/Link.hh>
#include <sdf/Material.hh>
#include <sdf/Model.hh>
#include <sdf/Sphere.hh>
#include <sdf/Visual.hh>


namespace sphere_network
{


// ============================================================
// GROUP / PATH
// ============================================================

struct Group
{
  std::string name;

  gz::math::Vector3d start;
  gz::math::Vector3d end;

  gz::math::Vector3d direction;

  double length{0.0};

  double baseSpeed{1.0};

  double minSpawnInterval{1.0};
  double maxSpawnInterval{4.0};

  double nextSpawnTime{0.0};
};


// ============================================================
// ACTIVE SPHERE
// ============================================================

struct ActiveSphere
{
  gz::sim::Entity entity{gz::sim::kNullEntity};

  std::string name;
  std::string groupName;

  gz::math::Vector3d start;
  gz::math::Vector3d direction;

  double speed{1.0};

  // Gazebo simulation time when sphere appeared.
  double spawnTime{0.0};

  // Full path distance.
  double pathLength{0.0};
};


// ============================================================
// PLUGIN
// ============================================================

class SphereNetworkSystem:
  public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate,
  public gz::sim::ISystemReset
{
  public:

    SphereNetworkSystem() = default;

    ~SphereNetworkSystem() override = default;


  // ==========================================================
  // CONFIGURE
  // ==========================================================

  public:

    void Configure(
      const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm,
      gz::sim::EventManager &_eventMgr
    ) override
    {
      // --------------------------------------------------------
      // Plugin is attached to the WORLD.
      // --------------------------------------------------------

      this->worldEntity = _entity;


      // --------------------------------------------------------
      // Direct Gazebo entity creator.
      //
      // This is the major architectural change.
      //
      // It works directly with Gazebo's ECM instead of calling:
      //
      // /world/city_test/create
      //
      // --------------------------------------------------------

      this->entityCreator =
        std::make_unique<gz::sim::SdfEntityCreator>(
          _ecm,
          _eventMgr
        );


      // --------------------------------------------------------
      // Global sphere settings
      // --------------------------------------------------------

      if (_sdf->HasElement("sphere_radius"))
      {
        this->sphereRadius =
          _sdf->Get<double>("sphere_radius");
      }


      if (_sdf->HasElement("speed_variation"))
      {
        this->speedVariation =
          _sdf->Get<double>("speed_variation");
      }


      if (_sdf->HasElement("motion_update_rate"))
      {
        this->motionUpdateRate =
          _sdf->Get<double>("motion_update_rate");
      }


      // --------------------------------------------------------
      // Random seed
      //
      // seed = 0:
      // different result each run
      //
      // fixed seed:
      // repeatable sphere sequence
      // --------------------------------------------------------

      unsigned int seed = 0;

      if (_sdf->HasElement("random_seed"))
      {
        seed =
          _sdf->Get<unsigned int>("random_seed");
      }


      if (seed == 0)
      {
        std::random_device rd;

        this->rng.seed(rd());
      }
      else
      {
        this->rng.seed(seed);
      }


      // --------------------------------------------------------
      // Read every GROUP from configuration.
      // --------------------------------------------------------

      if (!_sdf->HasElement("group"))
      {
        gzerr
          << "[SphereNetwork] No groups configured.\n";

        return;
      }


      auto groupElement =
      _sdf->FindElement("group");


      while (groupElement)
      {
        Group group;


        // ------------------------------------------------------
        // Name
        // ------------------------------------------------------

        group.name =
          groupElement->Get<std::string>("name");


        // ------------------------------------------------------
        // Start / end
        // ------------------------------------------------------

        group.start =
          groupElement->Get<gz::math::Vector3d>(
            "start"
          );


        group.end =
          groupElement->Get<gz::math::Vector3d>(
            "end"
          );


        // ------------------------------------------------------
        // Speed
        // ------------------------------------------------------

        group.baseSpeed =
          groupElement->Get<double>(
            "base_speed"
          );


        // ------------------------------------------------------
        // Spawn interval
        // ------------------------------------------------------

        group.minSpawnInterval =
          groupElement->Get<double>(
            "min_spawn_interval"
          );


        group.maxSpawnInterval =
          groupElement->Get<double>(
            "max_spawn_interval"
          );


        // ------------------------------------------------------
        // Calculate path
        // ------------------------------------------------------

        const gz::math::Vector3d delta =
          group.end -
          group.start;


        group.length =
          delta.Length();


        if (group.length <= 0.0001)
        {
          gzerr
            << "[SphereNetwork] Group ["
            << group.name
            << "] has zero-length path.\n";

          groupElement =
            groupElement->GetNextElement("group");

          continue;
        }


        group.direction =
          delta /
          group.length;


        // Will be initialised on first simulation update.
        group.nextSpawnTime = 0.0;


        this->groups.push_back(group);


        gzmsg
          << "[SphereNetwork] Loaded group ["
          << group.name
          << "] start="
          << group.start
          << " end="
          << group.end
          << " speed="
          << group.baseSpeed
          << " m/s\n";


        groupElement =
          groupElement->GetNextElement("group");
      }


      gzmsg
        << "[SphereNetwork] Ready. "
        << this->groups.size()
        << " groups loaded.\n";
    }


  // ==========================================================
  // PRE UPDATE
  // ==========================================================

  public:

    void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm
    ) override
    {
      // If Gazebo is paused, do nothing.
      if (_info.paused)
      {
        return;
      }


      const double now =
        std::chrono::duration<double>(
          _info.simTime
        ).count();


      // --------------------------------------------------------
      // Initialise group timers once.
      // --------------------------------------------------------

      if (!this->timersInitialised)
      {
        for (auto &group : this->groups)
        {
          group.nextSpawnTime =
            now +
            this->RandomSpawnDelay(group);
        }


        this->timersInitialised = true;
      }


      // --------------------------------------------------------
      // Spawn one sphere when each group's independent timer
      // expires.
      // --------------------------------------------------------

      for (auto &group : this->groups)
      {
        if (now >= group.nextSpawnTime)
        {
          this->SpawnSphere(
            group,
            now
          );


          group.nextSpawnTime =
            now +
            this->RandomSpawnDelay(group);
        }
      }


      // --------------------------------------------------------
      // Motion update
      // --------------------------------------------------------
      //
      // We don't need to command poses at Gazebo's full
      // 250 Hz physics rate.
      //
      // 60 Hz is smooth enough for visual targets while
      // keeping this scalable.
      // --------------------------------------------------------

      const double updatePeriod =
        1.0 /
        this->motionUpdateRate;


      if (
        now -
        this->lastMotionUpdate
        >= updatePeriod
      )
      {
        this->UpdateSpheres(
          now,
          _ecm
        );

        this->lastMotionUpdate =
          now;
      }
    }


  // ==========================================================
  // RESET
  // ==========================================================

  public:

    void Reset(
      const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &
    ) override
    {
      // Queue all existing spheres for removal.
      for (const auto &sphere : this->activeSpheres)
      {
        if (
          sphere.entity !=
          gz::sim::kNullEntity
        )
        {
          this->entityCreator->RequestRemoveEntity(
            sphere.entity
          );
        }
      }


      this->activeSpheres.clear();

      this->timersInitialised = false;

      this->lastMotionUpdate = 0.0;

      gzmsg
        << "[SphereNetwork] Reset.\n";
    }


  // ==========================================================
  // RANDOM SPAWN DELAY
  // ==========================================================

  private:

    double RandomSpawnDelay(
      const Group &_group
    )
    {
      std::uniform_real_distribution<double>
        distribution(
          _group.minSpawnInterval,
          _group.maxSpawnInterval
        );


      return distribution(
        this->rng
      );
    }


  // ==========================================================
  // RANDOM SPEED
  // ==========================================================

  private:

    double RandomSpeed(
      const Group &_group
    )
    {
      std::uniform_real_distribution<double>
        distribution(
          1.0 -
          this->speedVariation,

          1.0 +
          this->speedVariation
        );


      return
        _group.baseSpeed *
        distribution(this->rng);
    }


  // ==========================================================
  // RANDOM COLOUR
  // ==========================================================

  private: gz::math::Color RandomColour()
  {
    static const std::vector<gz::math::Color> colours =
    {
      // RED
      gz::math::Color(
        1.00, 0.00, 0.00, 1.00
      ),

      // GREEN
      gz::math::Color(
        0.00, 1.00, 0.10, 1.00
      ),

      // BLUE
      gz::math::Color(
        0.00, 0.25, 1.00, 1.00
      ),

      // YELLOW
      gz::math::Color(
        1.00, 1.00, 0.00, 1.00
      ),

      // PURPLE
      gz::math::Color(
        0.55, 0.05, 1.00, 1.00
      ),

      // ORANGE
      gz::math::Color(
        1.00, 0.35, 0.00, 1.00
      ),

      // BRIGHT PINK
      gz::math::Color(
        1.00, 0.05, 0.55, 1.00
      )
    };

    std::uniform_int_distribution<std::size_t> distribution(
      0,
      colours.size() - 1
    );

    return colours[
      distribution(this->rng)
    ];
  }

  // ==========================================================
  // SPAWN ONE SPHERE
  // ==========================================================

  private:

    void SpawnSphere(
      const Group &_group,
      const double _now
    )
    {
      // --------------------------------------------------------
      // Unique name
      // --------------------------------------------------------

      const std::string name =
        "sphere_" +
        _group.name +
        "_" +
        std::to_string(
          this->sphereCounter++
        );


      const double speed =
        this->RandomSpeed(
          _group
        );


      const gz::math::Color colour =
        this->RandomColour();


      // ========================================================
      // BUILD SPHERE USING SDFormat DOM
      // ========================================================


      // --------------------------------------------------------
      // Sphere geometry
      // --------------------------------------------------------

      sdf::Sphere sphereShape;

      sphereShape.SetRadius(
        this->sphereRadius
      );


      sdf::Geometry geometry;

      geometry.SetType(
        sdf::GeometryType::SPHERE
      );

      geometry.SetSphereShape(
        sphereShape
      );


      // --------------------------------------------------------
      // Material
      // --------------------------------------------------------

      sdf::Material material;

      material.SetAmbient(
        colour
      );

      material.SetDiffuse(
        colour
      );

      // Small emissive component makes the sphere colours
      // remain clear against dark / shadowed parts of the city.
      material.SetEmissive(
        gz::math::Color(  
          colour.R() * 0.20,
          colour.G() * 0.20,
          colour.B() * 0.20,
          1.0
        )
      );


      // --------------------------------------------------------
      // Visual
      // --------------------------------------------------------

      sdf::Visual visual;

      visual.SetName(
        "sphere_visual"
      );

      visual.SetGeom(
        geometry
      );

      visual.SetMaterial(
        material
      );


      // --------------------------------------------------------
      // Link
      //
      // No collision.
      // No inertia.
      // No physics.
      // --------------------------------------------------------

      sdf::Link link;

      link.SetName(
        "sphere_link"
      );

      link.AddVisual(
        visual
      );


      // --------------------------------------------------------
      // Model
      // --------------------------------------------------------

      sdf::Model modelSdf;

      modelSdf.SetName(
        name
      );


      // Static = no rigid-body physics.
      modelSdf.SetStatic(
        true
      );


      modelSdf.SetRawPose(
        gz::math::Pose3d(
          _group.start,
          gz::math::Quaterniond::Identity
        )
      );


      modelSdf.AddLink(
        link
      );


      // ========================================================
      // CREATE DIRECTLY IN ECM
      // ========================================================

      const gz::sim::Entity entity =
        this->entityCreator->CreateEntities(
          &modelSdf
        );


      this->entityCreator->SetParent(
        entity,
        this->worldEntity
      );


      // --------------------------------------------------------
      // Store internal state
      // --------------------------------------------------------

      ActiveSphere active;

      active.entity =
        entity;

      active.name =
        name;

      active.groupName =
        _group.name;

      active.start =
        _group.start;

      active.direction =
        _group.direction;

      active.speed =
        speed;

      active.spawnTime =
        _now;

      active.pathLength =
        _group.length;


      this->activeSpheres.push_back(
        active
      );


      gzmsg
        << "[SphereNetwork] Spawned ["
        << name
        << "] group="
        << _group.name
        << " speed="
        << speed
        << " m/s\n";
    }


  // ==========================================================
  // UPDATE ACTIVE SPHERES
  // ==========================================================

  private:

    void UpdateSpheres(
      const double _now,
      gz::sim::EntityComponentManager &_ecm
    )
    {
      std::size_t i = 0;


      while (
        i <
        this->activeSpheres.size()
      )
      {
        auto &sphere =
          this->activeSpheres[i];


        // ------------------------------------------------------
        // Distance travelled
        //
        // Calculated from absolute simulation time, so:
        //
        // - no accumulated integration error
        // - pauses work correctly
        // ------------------------------------------------------

        const double elapsed =
          _now -
          sphere.spawnTime;


        const double distance =
          elapsed *
          sphere.speed;


        // ------------------------------------------------------
        // End of path
        // ------------------------------------------------------

        if (
          distance >=
          sphere.pathLength
        )
        {
          // No service call.
          //
          // Gazebo queues deletion internally and processes
          // it during the simulation update.
          this->entityCreator->RequestRemoveEntity(
            sphere.entity
          );


          gzmsg
            << "[SphereNetwork] Deleted ["
            << sphere.name
            << "] at end of "
            << sphere.groupName
            << "\n";


          // Remove from active vector.
          this->activeSpheres.erase(
            this->activeSpheres.begin() +
            i
          );


          continue;
        }


        // ------------------------------------------------------
        // Position
        // ------------------------------------------------------

        const gz::math::Vector3d position =
          sphere.start +
          sphere.direction *
          distance;


        // ------------------------------------------------------
        // Direct Gazebo model pose command
        // ------------------------------------------------------

        gz::sim::Model model(
          sphere.entity
        );


        model.SetWorldPoseCmd(
          _ecm,

          gz::math::Pose3d(
            position,
            gz::math::Quaterniond::Identity
          )
        );


        ++i;
      }
    }


  // ==========================================================
  // DATA
  // ==========================================================

  private:

    gz::sim::Entity worldEntity{
      gz::sim::kNullEntity
    };


    std::unique_ptr<
      gz::sim::SdfEntityCreator
    > entityCreator;


    std::vector<Group> groups;

    std::vector<ActiveSphere>
      activeSpheres;


    std::mt19937 rng;


    std::size_t sphereCounter{0};


    double sphereRadius{0.30};

    // +/-10%
    double speedVariation{0.10};

    // Visual movement rate
    double motionUpdateRate{60.0};

    double lastMotionUpdate{0.0};

    bool timersInitialised{false};
};

}


// ============================================================
// REGISTER GAZEBO PLUGIN
// ============================================================

GZ_ADD_PLUGIN(
  sphere_network::SphereNetworkSystem,

  gz::sim::System,

  sphere_network::SphereNetworkSystem::ISystemConfigure,

  sphere_network::SphereNetworkSystem::ISystemPreUpdate,

  sphere_network::SphereNetworkSystem::ISystemReset
)


GZ_ADD_PLUGIN_ALIAS(
  sphere_network::SphereNetworkSystem,

  "sphere_network::SphereNetworkSystem"
)
