#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

class ImuUnitConverter : public rclcpp::Node
{
public:
  ImuUnitConverter() : Node("imu_unit_converter")
  {
    sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/livox/imu", 200,
      std::bind(&ImuUnitConverter::callback, this, std::placeholders::_1));

    pub_ = this->create_publisher<sensor_msgs::msg::Imu>(
      "/livox/imu_corrected", 200);

    RCLCPP_INFO(this->get_logger(), "IMU unit converter started");
  }

private:
  void callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    auto out = *msg;

    // Convert acceleration: g -> m/s^2
    constexpr double G = 9.81;
    out.linear_acceleration.x *= G;
    out.linear_acceleration.y *= G;
    out.linear_acceleration.z *= G;

    pub_->publish(out);
  }

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuUnitConverter>());
  rclcpp::shutdown();
  return 0;
}