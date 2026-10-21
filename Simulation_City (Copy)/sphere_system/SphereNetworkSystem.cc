// ============================================================
// SPHERE NETWORK SYSTEM
// ============================================================
//
//   - spawns individually
//   - receives a random colour
//   - receives a slightly random speed
//   - moves directly along its group path
//   - has no collision
//   - has no physics behaviour
//   - is deleted when it reaches the path end
//
// ============================================================

#include <atomic>
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

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double.pb.h>
#include <gz/transport/Node.hh>

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

// PINK LOOP ROUTE

struct PinkRoute
{
  bool enabled{false};

  std::vector<gz::math::Vector3d> waypoints;
  std::vector<double> segmentLengths;

  double totalLength{0.0};
  double speed{1.0};

  double distanceTravelled{0.0};
  double lastUpdateTime{0.0};

  gz::sim::Entity entity{gz::sim::kNullEntity};
  double spawnTime{0.0};
};

// GROUP

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

// ACTIVE SPHERE

struct ActiveSphere
{
  gz::sim::Entity entity{gz::sim::kNullEntity};

  std::string name;
  std::string groupName;

  gz::math::Vector3d start;
  gz::math::Vector3d direction;

  double speed{1.0};
  double spawnTime{0.0};
  double pathLength{0.0};
};

// PLUGIN

class SphereNetworkSystem:
  public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate,
  public gz::sim::ISystemReset
{
  public:
    SphereNetworkSystem() = default;
    ~SphereNetworkSystem() override = default;

  // CONFIG

  public:
    void Configure(
      const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm,
      gz::sim::EventManager &_eventMgr
    ) override
    {

      this->worldEntity = _entity;

      this->entityCreator =
        std::make_unique<gz::sim::SdfEntityCreator>(
          _ecm,
          _eventMgr
        );

      this->transportNode.Subscribe(
        "/sphere_network/start_traffic",
        &SphereNetworkSystem::OnStartTraffic,
        this
      );

      this->transportNode.Subscribe(
        "/sphere_network/start_pink",
        &SphereNetworkSystem::OnStartPink,
        this
      );

      this->transportNode.Subscribe(
        "/sphere_network/delete_all",
        &SphereNetworkSystem::OnDeleteAll,
        this
      );

      this->transportNode.Subscribe(
        "/sphere_network/pink_speed",
        &SphereNetworkSystem::OnPinkSpeed,
        this
      );

      // Global sphere settings

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

        group.name =
          groupElement->Get<std::string>("name");

        group.start =
          groupElement->Get<gz::math::Vector3d>(
            "start"
          );

        group.end =
          groupElement->Get<gz::math::Vector3d>(
            "end"
          );

        group.baseSpeed =
          groupElement->Get<double>(
            "base_speed"
          );

        group.minSpawnInterval =
          groupElement->Get<double>(
            "min_spawn_interval"
          );


        group.maxSpawnInterval =
          groupElement->Get<double>(
            "max_spawn_interval"
          );

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

      if (_sdf->HasElement("pink_route"))
      {
        auto pinkElement = _sdf->FindElement("pink_route");

        if (pinkElement->HasElement("speed"))
        {
          this->pinkRoute.speed = pinkElement->Get<double>("speed");
        }

        if (pinkElement->HasElement("waypoint"))
        {
          auto waypointElement = pinkElement->FindElement("waypoint");

          while (waypointElement)
          {
            this->pinkRoute.waypoints.push_back(
              waypointElement->Get<gz::math::Vector3d>()
            );

            waypointElement = waypointElement->GetNextElement("waypoint");
          }
        }

        if (this->pinkRoute.waypoints.size() < 2)
        {
          gzerr
            << "[SphereNetwork] pink_route needs at least two waypoints.\n";
        }
        else if (this->pinkRoute.speed <= 0.0)
        {
          gzerr
            << "[SphereNetwork] pink_route speed must be greater than zero.\n";
        }
        else
        {
          this->pinkRoute.totalLength = 0.0;
          this->pinkRoute.segmentLengths.clear();

          for (std::size_t i = 0; i < this->pinkRoute.waypoints.size(); ++i)
          {
            const std::size_t next =
              (i + 1) % this->pinkRoute.waypoints.size();

            const double segmentLength =
              (this->pinkRoute.waypoints[next] -
              this->pinkRoute.waypoints[i]).Length();

            this->pinkRoute.segmentLengths.push_back(segmentLength);
            this->pinkRoute.totalLength += segmentLength;
          }

          if (this->pinkRoute.totalLength <= 0.0001)
          {
            gzerr
              << "[SphereNetwork] pink_route has zero total length.\n";
          }
          else
          {
            this->pinkRoute.enabled = true;

            gzmsg
              << "[SphereNetwork] Loaded pink route with "
              << this->pinkRoute.waypoints.size()
              << " waypoints, speed="
              << this->pinkRoute.speed
              << " m/s, loop length="
              << this->pinkRoute.totalLength
              << " m.\n";
          }
        }
      }


      gzmsg
        << "[SphereNetwork] Ready. "
        << this->groups.size()
        << " groups loaded.\n";
    }

  // PRE UPDATE

  public:

    void PreUpdate(
      const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm
    ) override
    {
      if (_info.paused)
      {
        return;
      }

      const double now =
        std::chrono::duration<double>(
          _info.simTime
        ).count();

      // Delete / reset command

      if (this->deleteAllRequested.exchange(false))
      {
        this->DeleteAllSpheres();

        this->trafficRunning = false;
        this->timersInitialised = false;

        this->trafficStartRequested.store(false);
        this->pinkStartRequested.store(false);

        this->sphereCounter = 0;
        this->lastMotionUpdate = now;

        return;
      }

      // Live pink speed command

      if (this->pinkSpeedChangeRequested.exchange(false))
      {
        this->pinkRoute.speed =
          this->requestedPinkSpeed.load();

        gzmsg
          << "[SphereNetwork] Pink speed set to "
          << this->pinkRoute.speed
          << " m/s.\n";
      }

      // Start random traffic

      if (this->trafficStartRequested.exchange(false))
      {
        this->trafficRunning = true;
        this->timersInitialised = false;

        gzmsg
          << "[SphereNetwork] Traffic start command received.\n";
      }

      // Start dedicated pink target

      if (this->pinkStartRequested.exchange(false))
      {
        if (this->pinkRoute.enabled &&
            this->pinkRoute.entity == gz::sim::kNullEntity)
        {
          this->SpawnPinkSphere(now);
        }
      }

      // Initialise traffic timers only after traffic is started.

      if (this->trafficRunning &&
          !this->timersInitialised)
      {
        for (auto &group : this->groups)
        {
          group.nextSpawnTime =
            now +
            this->RandomSpawnDelay(group);
        }

        this->timersInitialised = true;
      }

      // Spawn random traffic only while enabled.

      if (this->trafficRunning)
      {
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
      }

      // Motion update

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

        this->UpdatePinkSphere(
          now,
          _ecm
        );

        this->lastMotionUpdate =
          now;
      }
    }

  // RESET

  public:

    void Reset(
      const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &
    ) override
    {
      this->DeleteAllSpheres();

      this->trafficRunning = false;
      this->timersInitialised = false;

      this->trafficStartRequested.store(false);
      this->pinkStartRequested.store(false);
      this->deleteAllRequested.store(false);
      this->pinkSpeedChangeRequested.store(false);

      this->sphereCounter = 0;
      this->lastMotionUpdate = 0.0;

      gzmsg
        << "[SphereNetwork] Reset.\n";
    }

  // TRANSPORT COMMAND CALLBACKS

  private:

    void OnStartTraffic(
      const gz::msgs::Boolean &_msg
    )
    {
      if (!_msg.data())
      {
        return;
      }

      this->trafficStartRequested.store(true);
    }

    void OnStartPink(
      const gz::msgs::Boolean &_msg
    )
    {
      if (!_msg.data())
      {
        return;
      }

      this->pinkStartRequested.store(true);
    }

    void OnDeleteAll(
      const gz::msgs::Boolean &_msg
    )
    {
      if (!_msg.data())
      {
        return;
      }

      this->deleteAllRequested.store(true);
    }

    void OnPinkSpeed(
      const gz::msgs::Double &_msg
    )
    {
      double speed =
        _msg.data();

      if (speed < 0.0)
      {
        speed = 0.0;
      }

      if (speed > 15.0)
      {
        speed = 15.0;
      }

      this->requestedPinkSpeed.store(speed);
      this->pinkSpeedChangeRequested.store(true);
    }

  // DELETE ALL SPHERES

  private:

    void DeleteAllSpheres()
    {
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

      if (
        this->pinkRoute.entity !=
        gz::sim::kNullEntity
      )
      {
        this->entityCreator->RequestRemoveEntity(
          this->pinkRoute.entity
        );

        this->pinkRoute.entity =
          gz::sim::kNullEntity;
      }

      this->pinkRoute.distanceTravelled = 0.0;
      this->pinkRoute.lastUpdateTime = 0.0;
      this->pinkRoute.spawnTime = 0.0;

      gzmsg
        << "[SphereNetwork] Deleted all spheres.\n";
    }

  // RANDOM SPAWN DELAY

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

  // RANDOM SPEED

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

  // RANDOM COLOUR

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
    };

    std::uniform_int_distribution<std::size_t> distribution(
      0,
      colours.size() - 1
    );

    return colours[
      distribution(this->rng)
    ];
  }

  // SPAWN ONE SPHERE

  private:

    void SpawnSphere(
      const Group &_group,
      const double _now
    )
    {
      // Unique name

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

      // BUILD SPHERE USING SDFormat DOM

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

      sdf::Material material;
      material.SetAmbient(
        colour
      );
      material.SetDiffuse(
        colour
      );
      material.SetEmissive(
        gz::math::Color(  
          colour.R() * 0.20,
          colour.G() * 0.20,
          colour.B() * 0.20,
          1.0
        )
      );

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

      sdf::Link link;

      link.SetName(
        "sphere_link"
      );
      link.AddVisual(
        visual
      );

      sdf::Model modelSdf;

      modelSdf.SetName(
        name
      );
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

      // CREATE DIRECTLY IN ECM

      const gz::sim::Entity entity =
        this->entityCreator->CreateEntities(
          &modelSdf
        );

      this->entityCreator->SetParent(
        entity,
        this->worldEntity
      );

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
  
  // SPAWN DEDICATED PINK SPHERE

  void SpawnPinkSphere(
    const double _now
  )
  {
    if (!this->pinkRoute.enabled ||
        this->pinkRoute.waypoints.empty())
    {
      return;
    }

    sdf::Sphere sphereShape;
    sphereShape.SetRadius(this->sphereRadius);

    sdf::Geometry geometry;
    geometry.SetType(sdf::GeometryType::SPHERE);
    geometry.SetSphereShape(sphereShape);

    const gz::math::Color pink(
      1.00, 0.05, 0.55, 1.00
    );

    sdf::Material material;
    material.SetAmbient(pink);
    material.SetDiffuse(pink);
    material.SetEmissive(
      gz::math::Color(0.20, 0.01, 0.11, 1.0)
    );

    sdf::Visual visual;
    visual.SetName("sphere_visual");
    visual.SetGeom(geometry);
    visual.SetMaterial(material);

    sdf::Link link;
    link.SetName("sphere_link");
    link.AddVisual(visual);

    sdf::Model modelSdf;
    modelSdf.SetName("pink_route_sphere");
    modelSdf.SetStatic(true);

    modelSdf.SetRawPose(
      gz::math::Pose3d(
        this->pinkRoute.waypoints.front(),
        gz::math::Quaterniond::Identity
      )
    );

    modelSdf.AddLink(link);

    this->pinkRoute.entity =
      this->entityCreator->CreateEntities(&modelSdf);

    this->entityCreator->SetParent(
      this->pinkRoute.entity,
      this->worldEntity
    );

    this->pinkRoute.spawnTime = _now;
    this->pinkRoute.distanceTravelled = 0.0;
    this->pinkRoute.lastUpdateTime = _now;

    gzmsg
      << "[SphereNetwork] Spawned dedicated pink route sphere.\n";
  }

  // UPDATE DEDICATED PINK SPHERE

  void UpdatePinkSphere(
    const double _now,
    gz::sim::EntityComponentManager &_ecm
  )
  {
    if (!this->pinkRoute.enabled ||
        this->pinkRoute.entity == gz::sim::kNullEntity ||
        this->pinkRoute.totalLength <= 0.0001)
    {
      return;
    }

    double dt =
      _now -
      this->pinkRoute.lastUpdateTime;

    if (dt < 0.0)
    {
      dt = 0.0;
    }

    this->pinkRoute.lastUpdateTime =
      _now;

    this->pinkRoute.distanceTravelled +=
      dt *
      this->pinkRoute.speed;

    this->pinkRoute.distanceTravelled =
      std::fmod(
        this->pinkRoute.distanceTravelled,
        this->pinkRoute.totalLength
      );

    double distance =
      this->pinkRoute.distanceTravelled;

    gz::math::Vector3d position =
      this->pinkRoute.waypoints.front();

    for (
      std::size_t i = 0;
      i < this->pinkRoute.segmentLengths.size();
      ++i
    )
    {
      const double segmentLength =
        this->pinkRoute.segmentLengths[i];

      if (segmentLength <= 0.0001)
      {
        continue;
      }

      if (distance <= segmentLength)
      {
        const std::size_t next =
          (i + 1) %
          this->pinkRoute.waypoints.size();

        const double t =
          distance /
          segmentLength;

        position =
          this->pinkRoute.waypoints[i] +
          (
            this->pinkRoute.waypoints[next] -
            this->pinkRoute.waypoints[i]
          ) * t;

        break;
      }

      distance -=
        segmentLength;
    }

    gz::sim::Model model(
      this->pinkRoute.entity
    );

    model.SetWorldPoseCmd(
      _ecm,
      gz::math::Pose3d(
        position,
        gz::math::Quaterniond::Identity
      )
    );
  }

  // UPDATE ACTIVE SPHERES

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

        const double elapsed =
          _now -
          sphere.spawnTime;


        const double distance =
          elapsed *
          sphere.speed;

        // End of path

        if (
          distance >=
          sphere.pathLength
        )
        {
          this->entityCreator->RequestRemoveEntity(
            sphere.entity
          );

          gzmsg
            << "[SphereNetwork] Deleted ["
            << sphere.name
            << "] at end of "
            << sphere.groupName
            << "\n";

          this->activeSpheres.erase(
            this->activeSpheres.begin() +
            i
          );

          continue;
        }

        const gz::math::Vector3d position =
          sphere.start +
          sphere.direction *
          distance;

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

  // DATA

  private:

    gz::sim::Entity worldEntity{
      gz::sim::kNullEntity
    };

    std::unique_ptr<
      gz::sim::SdfEntityCreator
    > entityCreator;

    std::vector<Group> groups;

    PinkRoute pinkRoute;

    std::vector<ActiveSphere>
      activeSpheres;

    std::mt19937 rng;

    gz::transport::Node transportNode;

    std::atomic<bool> trafficStartRequested{false};
    std::atomic<bool> pinkStartRequested{false};
    std::atomic<bool> deleteAllRequested{false};

    std::atomic<double> requestedPinkSpeed{6.0};
    std::atomic<bool> pinkSpeedChangeRequested{false};

    bool trafficRunning{false};
    std::size_t sphereCounter{0};
    double sphereRadius{0.30};
    double speedVariation{0.10};
    double motionUpdateRate{60.0};
    double lastMotionUpdate{0.0};
    bool timersInitialised{false};
};

}

// REGISTER GAZEBO PLUGIN

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
